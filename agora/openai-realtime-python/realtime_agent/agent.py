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
from .realtime.struct import ErrorMessage, FunctionCallOutputItemParam, InputAudioBufferCommitted, InputAudioBufferSpeechStarted, InputAudioBufferSpeechStopped, InputAudioTranscription, ItemCreate, ItemCreated, ItemInputAudioTranscriptionCompleted, RateLimitsUpdated, ResponseAudioDelta, ResponseAudioDone, ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone, ResponseContentPartAdded, ResponseContentPartDone, ResponseCreate, ResponseCreateParams, ResponseCreated, ResponseDone, ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone, ResponseOutputItemAdded, ResponseOutputItemDone, ServerVADUpdateParams, SessionUpdate, SessionUpdateParams, SessionUpdated, Voices, to_json, ItemParam, UserMessageItemParam
from .realtime.connection import RealtimeApiConnection
from .tools import ClientToolCallResponse, ToolContext
from .utils import PCMWriter
# conversation_log = []

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
        print(options)
        channel = engine.create_channel(options)
        
        # Configure audio settings
        options.sample_rate = 24000  # Match model output sample rate
        options.channels = 1  # Mono audio
        options.audio_format = "pcm16"  # PCM16 format
        
        await channel.connect()

        try:
            async with RealtimeApiConnection(
                base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
                api_key=os.getenv("OPENAI_API_KEY"),
                verbose=False,
            ) as connection:
                await connection.send_request(
                    SessionUpdate(
                        session=SessionUpdateParams(
                            # MARK: check this
                            turn_detection=inference_config.turn_detection,
                            tools=tools.model_description() if tools else [],
                            tool_choice="auto",
                            input_audio_format="pcm16",
                            output_audio_format="pcm16",
                            instructions=inference_config.system_message,
                            voice=inference_config.voice,
                            model=os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview"),
                            # model=os.environ.get("OPENAI_MODEL"),
                            modalities=["text", "audio"],
                            temperature=0.6,
                            max_response_output_tokens="inf",
                            input_audio_transcription=InputAudioTranscription(model="whisper-1")
                        )
                    )
                )

                start_session_message = await anext(connection.listen())
                # assert isinstance(start_session_message, messages.StartSession)
                if isinstance(start_session_message, SessionUpdated):
                    logger.info(
                        f"Session started: {start_session_message.session.id} model: {start_session_message.session.model}"
                    )
                elif isinstance(start_session_message, ErrorMessage):
                    logger.info(
                        f"Error: {start_session_message.error}"
                    )

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

    async def stop(self):
        try:
            logger.info("Stopping agent")
            await self.channel.disconnect()
            logger.info("Disconnected")
        except Exception as e:
            logger.error(f"Error: {e}")

        # Add any additional cleanup logic here if needed
        self._running = False


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
            log_file_path = "conversation_log.json"
            if os.path.exists(log_file_path):
                with open(log_file_path, "w") as f:
                    json.dump([], f)
                logger.info("Cleared conversation_log.json for new session.")
            self.subscribe_user = await wait_for_remote_user(self.channel)
            logger.info(f"Subscribing to user {self.subscribe_user}")
            await self.channel.subscribe_audio(self.subscribe_user)

            # # Send welcome message
            # await self.connection.send_request(
            #     ResponseCreate(
            #         response=ResponseCreateParams(
            #             input_items=[
            #                 UserMessageItemParam(
            #                     content=[{"type": "text", "text": "Hello! I'm your AI interviewer. Can you please tell me about yourself?"}]
            #                 )
            #             ]
            #         )
            #     )
            # )

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if self.subscribe_user == user_id:
                    self.subscribe_user = None
                    logger.info("Subscribed user left, disconnecting")
                    await self.channel.disconnect()

            self.channel.on("user_left", on_user_left)

            disconnected_future = asyncio.Future[None]()

            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)

            self.channel.on("connection_state_changed", callback)

            asyncio.create_task(self.rtc_to_model()).add_done_callback(log_exception)
            asyncio.create_task(self.model_to_rtc()).add_done_callback(log_exception)

            asyncio.create_task(self._process_model_messages()).add_done_callback(
                log_exception
            )

            await disconnected_future
            logger.info("Agent finished running")
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    async def rtc_to_model(self) -> None:
        logger.info("Starting rtc_to_model task")
        while self.subscribe_user is None or self.channel.get_audio_frames(self.subscribe_user) is None:
            logger.info(f"Waiting for subscribe_user: {self.subscribe_user}, audio_frames: {self.channel.get_audio_frames(self.subscribe_user) is not None}")
            await asyncio.sleep(0.1)

        audio_frames = self.channel.get_audio_frames(self.subscribe_user)
        logger.info(f"Got audio frames for user {self.subscribe_user}")

        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)

        try:
            async for audio_frame in audio_frames:
                # Process received audio (send to model)
                _monitor_queue_size(self.audio_queue, "audio_queue")
                logger.info(f"Received audio frame from RTC, size: {len(audio_frame.data)} bytes")
                await self.connection.send_audio_data(audio_frame.data)
                logger.info("Sent audio data to model")

                # Write PCM data if enabled
                await pcm_writer.write(audio_frame.data)

                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the exception to propagate cancellation
        except Exception as e:
            logger.error(f"Error in rtc_to_model: {e}")
            raise

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)
        logger.info("Starting model_to_rtc task")

        try:
            while True:
                # Get audio frame from the model output
                logger.info("Waiting for audio frame from queue")
                frame = await self.audio_queue.get()
                logger.info(f"Received audio frame of size: {len(frame)} bytes")

                # Process sending audio (to RTC)
                logger.info("Pushing audio frame to RTC channel")
                await self.channel.push_audio_frame(frame)
                logger.info("Audio frame pushed to RTC channel")

                # Write PCM data if enabled
                await pcm_writer.write(frame)

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task
        except Exception as e:
            logger.error(f"Error in model_to_rtc: {e}")
            raise

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
        logger.info("Starting _process_model_messages task")
        async for message in self.connection.listen():
            # logger.info(f"Received message {message=}")
            logger.info(f"Received message type: {type(message).__name__}")
            match message:
                case ResponseAudioDelta():
                    # logger.info("Received audio message")
                    logger.info(f"Received audio delta: response_id:{message.response_id}, item_id:{message.item_id}, delta length:{len(message.delta)}")
                    self.audio_queue.put_nowait(base64.b64decode(message.delta))
                    # loop.call_soon_threadsafe(self.audio_queue.put_nowait, base64.b64decode(message.delta))
                    logger.debug(f"TMS:ResponseAudioDelta: response_id:{message.response_id},item_id: {message.item_id}")
                case ResponseAudioTranscriptDelta():
                    # logger.info(f"Received text message {message=}")
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

                    response_schemas = [
                        ResponseSchema(name="ai", description="given the response from the ai"),
                        ResponseSchema(
                            name="aicontext",
                            description="you need to understand the response from the ai and need to respond with true if the interview is ended and false if the interview is still going on",
                        ),
                    ]
                    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

                    format_instructions = output_parser.get_format_instructions()
                    question=self.last_ai_question
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
                        # await self.channel.disconnect()  # Correct way to leave
                        await self.stop()  # Stop agent loop
                        return

                case InputAudioBufferSpeechStarted():
                    await self.channel.clear_sender_audio_buffer()
                    # clear the audio queue so audio stops playing
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info(f"TMS:InputAudioBufferSpeechStarted: item_id: {message.item_id}")
                case InputAudioBufferSpeechStopped():
                    logger.info(f"TMS:InputAudioBufferSpeechStopped: item_id: {message.item_id}")
                    pass
                case ItemInputAudioTranscriptionCompleted():
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
                    
                    with open("conversation_log.json", "w") as f:
                        json.dump(self.conversation_log, f, indent=4)
                    

                #  InputAudioBufferCommitted
                case InputAudioBufferCommitted():
                    pass
                case ItemCreated():
                    pass
                # ResponseCreated
                case ResponseCreated():
                    pass
                # ResponseDone
                case ResponseDone():
                    pass

                # ResponseOutputItemAdded
                case ResponseOutputItemAdded():
                    pass

                # ResponseContenPartAdded
                case ResponseContentPartAdded():
                    pass
                # ResponseAudioDone
                case ResponseAudioDone():
                    pass
                # ResponseContentPartDone
                case ResponseContentPartDone():
                    pass
                # ResponseOutputItemDone
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


