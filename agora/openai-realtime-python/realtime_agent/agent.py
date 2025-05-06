import asyncio
import base64
import logging
import os
import json
from datetime import datetime
from builtins import anext
from typing import Any

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from attr import dataclass

from agora_realtime_ai_api.rtc import Channel, ChatMessage, RtcEngine, RtcOptions

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .logger import setup_logger
from .realtime.struct import (
    ErrorMessage, FunctionCallOutputItemParam, InputAudioBufferAppend, 
    InputAudioBufferCommitted, InputAudioBufferSpeechStarted, 
    InputAudioBufferSpeechStopped, InputAudioTranscription, 
    ItemCreate, ItemCreated, ItemInputAudioTranscriptionCompleted, 
    RateLimitsUpdated, ResponseAudioDelta, ResponseAudioDone, 
    ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone, 
    ResponseContentPartAdded, ResponseContentPartDone, ResponseCreate, 
    ResponseCreated, ResponseDone, ResponseFunctionCallArgumentsDelta, 
    ResponseFunctionCallArgumentsDone, ResponseOutputItemAdded, 
    ResponseOutputItemDone, ServerVADUpdateParams, SessionUpdate, 
    SessionUpdateParams, SessionUpdated, UserMessageItemParam, Voices, to_json
)
from .realtime.connection import RealtimeApiConnection
from .tools import ClientToolCallResponse, ToolContext
from .utils import PCMWriter
from .realtime.google_stt import transcribe_audio

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

def _monitor_queue_size(queue: asyncio.Queue, queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logger.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel) -> int:
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout of 30 seconds
        remote_user = await asyncio.wait_for(future, timeout=15.0)
        return remote_user
    except KeyboardInterrupt:
        future.cancel()
        
    except Exception as e:
        logger.error(f"Error waiting for remote user: {e}")
        raise


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    system_message: str | None = None
    turn_detection: ServerVADUpdateParams | None = None  # MARK: CHECK!
    voice: Voices | None = None


class RealtimeKitAgent:
    engine: RtcEngine
    channel: Channel
    connection: RealtimeApiConnection
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    message_queue: asyncio.Queue[ResponseAudioTranscriptDelta] = (
        asyncio.Queue()
    )
    message_done_queue: asyncio.Queue[ResponseAudioTranscriptDone] = (
        asyncio.Queue()
    )
    tools: ToolContext | None = None

    _client_tool_futures: dict[str, asyncio.Future[ClientToolCallResponse]]

    @classmethod
    async def setup_and_run_agent(
        cls,
        *,
        engine: RtcEngine,
        options: RtcOptions,
        inference_config: InferenceConfig,
        tools: ToolContext | None,
    ) -> None:
        channel = engine.create_channel(options)
        await channel.connect()

        try:
            async with RealtimeApiConnection(
                base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
                api_key=os.getenv("OPENAI_API_KEY"),
                verbose=False,
            ) as connection:
                # Modified: Removed input_audio_transcription to let Google STT handle all transcription
                await connection.send_request(
                    SessionUpdate(
                        session=SessionUpdateParams(
                            turn_detection=inference_config.turn_detection,
                            tools=tools.model_description() if tools else [],
                            tool_choice="auto",
                            input_audio_format="pcm16",
                            output_audio_format="pcm16",
                            instructions=inference_config.system_message,
                            voice=inference_config.voice,
                            model=os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview"),
                            modalities=["text", "audio"],
                            temperature=0.6,
                            max_response_output_tokens="inf",
                            # Removed Whisper-1 transcription configuration
                        )
                    )
                )

                start_session_message = await anext(connection.listen())
                # Check session status
                if isinstance(start_session_message, SessionUpdated):
                    logger.info(
                        f"Session started: {start_session_message.session.id} model: {start_session_message.session.model}"
                    )
                elif isinstance(start_session_message, ErrorMessage):
                    logger.error(
                        f"Error starting session: {start_session_message.error}"
                    )
                    return  # Exit if session creation failed

                agent = cls(
                    connection=connection,
                    tools=tools,
                    channel=channel,
                )
                await agent.run()

        finally:
            await channel.disconnect()
            await connection.close()

    def __init__(
        self,
        *,
        connection: RealtimeApiConnection,
        tools: ToolContext | None,
        channel: Channel,
    ) -> None:
        self.connection = connection
        self.tools = tools
        self._client_tool_futures = {}
        self.channel = channel
        self.subscribe_user = None
        self.write_pcm = os.environ.get("WRITE_AGENT_PCM", "false") == "true"
        logger.info(f"Write PCM: {self.write_pcm}")
        self.conversation_log = []
        self.last_ai_question = ""
        self.last_user_answer = ""
        self._running = True

    async def stop(self):
        try:
            logger.info("Stopping agent")
            self._running = False
            await self.channel.disconnect()
            logger.info("Disconnected")
        except Exception as e:
            logger.error(f"Error stopping agent: {e}")

    async def run(self) -> None:
        try:
            def log_exception(t: asyncio.Task[Any]) -> None:
                if not t.cancelled() and t.exception():
                    logger.error(
                        "unhandled exception",
                        exc_info=t.exception(),
                    )

            def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
                logger.info(f"Received stream message with length: {length}")

            self.channel.on("stream_message", on_stream_message)

            logger.info("Waiting for remote user to join")
            # Clear conversation log file for new session
            log_file_path = "conversation_log.json"
            if os.path.exists(log_file_path):
                with open(log_file_path, "w") as f:
                    json.dump([], f)
                logger.info("Cleared conversation_log.json for new session.")
                
            # Wait for user to join
            self.subscribe_user = await wait_for_remote_user(self.channel)
            logger.info(f"User joined! Subscribing to user {self.subscribe_user}")
            await self.channel.subscribe_audio(self.subscribe_user)

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if self.subscribe_user == user_id:
                    self.subscribe_user = None
                    logger.info("Subscribed user left, disconnecting")
                    await self.stop()

            self.channel.on("user_left", on_user_left)

            disconnected_future = asyncio.Future[None]()

            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:  # Disconnected
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)

            self.channel.on("connection_state_changed", callback)

            # Start processing tasks
            audio_task = asyncio.create_task(self.rtc_to_model())
            audio_task.add_done_callback(log_exception)
            
            rtc_task = asyncio.create_task(self.model_to_rtc())
            rtc_task.add_done_callback(log_exception)
            
            msg_task = asyncio.create_task(self._process_model_messages())
            msg_task.add_done_callback(log_exception)

            # Wait for disconnection
            await disconnected_future
            logger.info("Agent finished running")
            
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    async def rtc_to_model(self) -> None:
        """Send audio from RTC to model"""
        # Set _running to True to start processing immediately
        self._running = True
            
        while self.subscribe_user is None or self.channel.get_audio_frames(self.subscribe_user) is None:
            if not self._running:
                return
            await asyncio.sleep(0.1)

        audio_frames = self.channel.get_audio_frames(self.subscribe_user)

        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)
        
        # Buffer to accumulate audio for batch processing with Google STT
        audio_buffer = bytearray()
        last_transcription_time = 0
        transcription_interval = 1.0  # Process transcriptions every 1 second

        try:
            async for audio_frame in audio_frames:
                if not self._running:
                    break
                    
                # Send audio to OpenAI for processing via the connection
                await self.connection.send_audio_data(audio_frame.data)
                
                # Accumulate audio for Google STT
                audio_buffer.extend(audio_frame.data)
                
                # Check if it's time to process the accumulated audio with Google STT
                current_time = asyncio.get_event_loop().time()
                if current_time - last_transcription_time >= transcription_interval and len(audio_buffer) > 0:
                    # Get a copy of the current buffer and clear it
                    current_buffer = bytes(audio_buffer)
                    audio_buffer.clear()
                    
                    # Process in background task to not block audio processing
                    asyncio.create_task(self._process_transcription(current_buffer))
                    last_transcription_time = current_time

                # Write PCM data if enabled
                await pcm_writer.write(audio_frame.data)
                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the exception to propagate cancellation

    async def _process_transcription(self, audio_data: bytes) -> None:
        """Process audio data with Google STT and send transcription to the agent"""
        try:
            # Process the audio with Google STT
            transcript = transcribe_audio(audio_data)
            if transcript and transcript.strip():
                logger.info(f"Google STT transcript: {transcript}")
                
                # Send the transcript as a user message to the agent
                await self.connection.send_request(ItemCreate(
                    item=UserMessageItemParam(
                        content=[{"type": "text", "text": transcript}]
                    )
                ))
                
                # Trigger AI response
                await self.connection.send_request(ResponseCreate())
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)

        try:
            while self._running:
                # Get audio frame from the model output
                try:
                    frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                    # Process sending audio (to RTC)
                    await self.channel.push_audio_frame(frame)
                    # Write PCM data if enabled
                    await pcm_writer.write(frame)
                except asyncio.TimeoutError:
                    continue  # Just continue the loop if timeout

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task

    async def handle_funtion_call(self, message: ResponseFunctionCallArgumentsDone) -> None:
        function_call_response = await self.tools.execute_tool(message.name, message.arguments)
        logger.info(f"Function call response: {function_call_response}")
        await self.connection.send_request(
            ItemCreate(
                item = FunctionCallOutputItemParam(
                    call_id=message.call_id,
                    output=function_call_response.json_encoded_output
                )
            )
        )
        await self.connection.send_request(
            ResponseCreate()
        )

    async def _process_model_messages(self) -> None:
        async for message in self.connection.listen():
            if not self._running:
                break
                
            # logger.info(f"Received message {message=}")
            match message:
                case ResponseAudioDelta():
                    # logger.info("Received audio message")
                    self.audio_queue.put_nowait(base64.b64decode(message.delta))
                    logger.debug(f"TMS:ResponseAudioDelta: response_id:{message.response_id},item_id: {message.item_id}")
                    
                case ResponseAudioTranscriptDelta():
                    # Send transcript updates to chat
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))

                case ResponseAudioTranscriptDone():
                    logger.info(f"Text message done: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                    print(f"ai: {message.transcript}")
                    self.last_ai_question = message.transcript

                    # Check if interview is completed
                    response_schemas = [
                        ResponseSchema(name="ai", description="given the response from the ai"),
                        ResponseSchema(
                            name="aicontext",
                            description="you need to understand the response from the ai and need to respond with true if the interview is ended and false if the interview is still going on",
                        ),
                    ]
                    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

                    format_instructions = output_parser.get_format_instructions()
                    question = self.last_ai_question
                    prompt = PromptTemplate(
                        template="understand the ai response as best as possible.\n{format_instructions}\n{question}",
                        input_variables=["question"],
                        partial_variables={"format_instructions": format_instructions},
                    )
                    model = ChatOpenAI(temperature=0)
                    chain = prompt | model | output_parser

                    x=chain.invoke({"question": f"{question}"})
                    print(x['aicontext'])

                    if x['aicontext']=='true':
                        logger.info("Interview concluded. Bot will now leave the channel.")
                        
                        # Save log
                        with open("conversation_log.json", "w") as f:
                            json.dump(self.conversation_log, f, indent=4)

                        # Gracefully disconnect
                        await self.stop()
                        return

                case InputAudioBufferSpeechStarted():
                    await self.channel.clear_sender_audio_buffer()
                    # Clear the audio queue so audio stops playing
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info(f"TMS:InputAudioBufferSpeechStarted: item_id: {message.item_id}")
                    
                case InputAudioBufferSpeechStopped():
                    logger.info(f"TMS:InputAudioBufferSpeechStopped: item_id: {message.item_id}")
                    
                case ItemInputAudioTranscriptionCompleted():
                    # This is now only used for handling transcription results from OpenAI
                    # but we'll keep it as a backup path for handling transcription results
                    logger.info(f"ItemInputAudioTranscriptionCompleted: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                    print(f"user: {message.transcript}")
                    self.last_user_answer = message.transcript
                    if self.last_user_answer:
                        self.conversation_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "question": self.last_ai_question,
                            "answer": self.last_user_answer
                        })

                    self.last_user_answer = ""
                    
                    # Save conversation log after each exchange
                    with open("conversation_log.json", "w") as f:
                        json.dump(self.conversation_log, f, indent=4)
                
                # Add handler for ItemCreated to handle user messages created through Google STT
                case ItemCreated() as item_created:
                    if hasattr(item_created.item, 'role') and item_created.item.role == 'user':
                        if hasattr(item_created.item, 'content') and len(item_created.item.content) > 0:
                            for content_part in item_created.item.content:
                                if content_part.get('type') == 'text':
                                    user_text = content_part.get('text', '')
                                    logger.info(f"User message from Google STT: {user_text}")
                                    print(f"user: {user_text}")
                                    
                                    self.last_user_answer = user_text
                                    if self.last_user_answer:
                                        self.conversation_log.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "question": self.last_ai_question,
                                            "answer": self.last_user_answer
                                        })
                                    
                                    self.last_user_answer = ""
                                    
                                    # Save conversation log after each exchange
                                    with open("conversation_log.json", "w") as f:
                                        json.dump(self.conversation_log, f, indent=4)
                                    
                                    # Trigger AI response after receiving user input
                                    await self.connection.send_request(ResponseCreate())
                
                case InputAudioBufferCommitted():
                    pass
                case ResponseCreated():
                    pass
                case ResponseDone():
                    pass
                case ResponseOutputItemAdded():
                    pass
                case ResponseContentPartAdded():
                    pass
                case ResponseAudioDone():
                    pass
                case ResponseContentPartDone():
                    pass
                case ResponseOutputItemDone():
                    pass
                case SessionUpdated():
                    pass
                case RateLimitsUpdated():
                    pass
                case ResponseFunctionCallArgumentsDone():
                    asyncio.create_task(
                        self.handle_funtion_call(message)
                    )
                case ResponseFunctionCallArgumentsDelta():
                    pass
                case _:
                    logger.warning(f"Unhandled message {message=}")