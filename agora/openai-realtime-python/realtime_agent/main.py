# Function to run the agent in a new process
import asyncio
import logging
import os
import signal
from multiprocessing import Process

from aiohttp import web
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from realtime_agent.realtime.tools_example import AgentTools

from .realtime.struct import PCM_CHANNELS, PCM_SAMPLE_RATE, ServerVADUpdateParams, Voices

from .agent import InferenceConfig, RealtimeKitAgent
from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions
from .logger import setup_logger
from .parse_args import parse_args, parse_args_realtimekit

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

load_dotenv(override=True)
app_id = os.environ.get("AGORA_APP_ID")
app_cert = os.environ.get("AGORA_APP_CERT")

if not app_id:
    raise ValueError("AGORA_APP_ID must be set in the environment.")


class StartAgentRequestBody(BaseModel):
    channel_name: str = Field(..., description="The name of the channel")
    uid: int = Field(..., description="The UID of the user")
    language: str = Field("en", description="The language of the agent")
    system_instruction: str = Field("", description="The system instruction for the agent")
    voice: str = Field("alloy", description="The voice of the agent")
    token: str = Field(..., description="The token of the user")


class StopAgentRequestBody(BaseModel):
    channel_name: str = Field(..., description="The name of the channel")


# Function to monitor the process and perform extra work when it finishes
async def monitor_process(channel_name: str, process: Process):
    # Wait for the process to finish in a non-blocking way
    await asyncio.to_thread(process.join)

    logger.info(f"Process for channel {channel_name} has finished")

    # Perform additional work after the process finishes
    # For example, removing the process from the active_processes dictionary
    if channel_name in active_processes:
        active_processes.pop(channel_name)

    # Perform any other cleanup or additional actions you need here
    logger.info(f"Cleanup for channel {channel_name} completed")

    logger.info(f"Remaining active processes: {len(active_processes.keys())}")

def handle_agent_proc_signal(signum, frame):
    logger.info(f"Agent process received signal {signal.strsignal(signum)}. Exiting...")
    os._exit(0)


def run_agent_in_process(
    engine_app_id: str,
    engine_app_cert: str,
    channel_name: str,
    uid: str,
    inference_config: InferenceConfig,
    token: str,
):  # Set up signal forwarding in the child process
    signal.signal(signal.SIGINT, handle_agent_proc_signal)  # Forward SIGINT
    signal.signal(signal.SIGTERM, handle_agent_proc_signal)  # Forward SIGTERM
    asyncio.run(
        RealtimeKitAgent.setup_and_run_agent(
            engine=RtcEngine(appid=engine_app_id, appcert=engine_app_cert),
            options=RtcOptions(
                channel_name=channel_name,
                uid=uid,
                sample_rate=PCM_SAMPLE_RATE,
                channels=PCM_CHANNELS,
                # enable_pcm_dump= os.environ.get("WRITE_RTC_PCM", "false") == "true"
            ),
            inference_config=inference_config,
            tools=None,
            # tools=AgentTools() # tools example, replace with this line
        )
    )


# HTTP Server Routes
async def start_agent(request):
    try:
        # Parse and validate JSON body using the pydantic model
        try:
            data = await request.json()
            validated_data = StartAgentRequestBody(**data)
        except ValidationError as e:
            return web.json_response(
                {"error": "Invalid request data", "details": e.errors()}, status=400
            )

        # Parse JSON body
        channel_name = validated_data.channel_name
        uid = validated_data.uid
        language = validated_data.language
        system_instruction = validated_data.system_instruction
        voice = validated_data.voice
        token = validated_data.token

        # Check if a process is already running for the given channel_name
        if (
            channel_name in active_processes
            and active_processes[channel_name].is_alive()
        ):
            return web.json_response(
                {"error": f"Agent already running for channel: {channel_name}"},
                status=400,
            )
        num_questions=5
        job_description= "Full Stack Developer with expertise in javascript and Data Structures & Algorithms"
        prioritized_skills="javascript, DSA, Data Structures, Algorithms"
        remaining_skills="Backend Development, Algorithm Design, Problem Solving, System Design, Django, Flask"
        difficulty_level="medium"
        duration_minutes=5
        system_message = ""
        if language == "en":
            system_message = f"""
            <s>[INST]
 As an AI interviewer, generate only {num_questions} technical interview questions for a candidate in english, Do not anser back to the user, don't correct him behave like a professional interviewer, Ask the questions one by one let the user anser the questions. The followup questions can be based on the user.
    
    Job Description:
    {job_description}
    
    Priority Topics to Cover:
    {', '.join(prioritized_skills)}
    
    Additional Skills (if needed):
    {', '.join(remaining_skills)}
    
    Difficulty Level: {difficulty_level.upper()}
    Interview Duration: {duration_minutes} minutes
    
    Requirements:
    1. MUST generate questions covering the Priority Topics first
    2. Only use Additional Skills if more questions are needed
    3. Each question should clearly indicate which skill/topic it covers
    4. Do not anser back to the user, don't correct him behave like a professional interviewer
    5. Do not give any explanation for the answer provided by the user just ask him if he wants to confirm the answer and move on to the next question.
    **You are NOT allowed to:**
    - Break character.
    - Provide solutions or answers to the user.
    - Respond casually or use humor.
    ** strictly follow the instructions below**
    - Strictly behave like an interviwer
    - strictly follow the instructions provided
    - Don't cross the instructions in any kind of situation
    Generate questions that:
    1. Are specific to the required skills
    2. Test both theoretical knowledge and practical experience
    3. Include a mix of technical and problem-solving questions
    4. Are clear and unambiguous
    5. Match the {difficulty_level} difficulty level
    6. Can be reasonably answered within the time constraints
    
    For {difficulty_level.upper()} level:
    - Beginner: Focus on fundamental concepts and basic implementations
    - Intermediate: Include practical scenarios and common challenges
    - Advanced: Cover complex scenarios and best practices
    - Expert: Focus on architecture decisions, trade-offs, and cutting-edge concepts
    
    Format each question with:
    1. The main question
    2. Expected key points in the answer
    3. Follow-up questions based on possible responses
    4. Estimated time for discussion (in minutes)
    5. Primary skill/topic being assessed
    """

        if system_instruction:
            system_message = system_instruction

        if voice not in Voices.__members__.values():
            return web.json_response(
                {"error": f"Invalid voice: {voice}."},
                status=400,
            )

        inference_config = InferenceConfig(
            system_message=system_message,
            voice=voice,
            turn_detection=ServerVADUpdateParams(
                type="server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=200
            ),
        )
        # Create a new process for running the agent
        process = Process(
            target=run_agent_in_process,
            args=(app_id, app_cert, channel_name, uid, inference_config, token),
        )

        try:
            process.start()
        except Exception as e:
            logger.error(f"Failed to start agent process: {e}")
            return web.json_response(
                {"error": f"Failed to start agent: {e}"}, status=500
            )

        # Store the process in the active_processes dictionary using channel_name as the key
        active_processes[channel_name] = process

        # Monitor the process in a background asyncio task
        asyncio.create_task(monitor_process(channel_name, process))

        return web.json_response({"status": "Agent started!"})

    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        return web.json_response({"error": str(e)}, status=500)


# HTTP Server Routes: Stop Agent
async def stop_agent(request):
    try:
        # Parse and validate JSON body using the pydantic model
        try:
            data = await request.json()
            validated_data = StopAgentRequestBody(**data)
        except ValidationError as e:
            return web.json_response(
                {"error": "Invalid request data", "details": e.errors()}, status=400
            )

        # Parse JSON body
        channel_name = validated_data.channel_name

        # Find and terminate the process associated with the given channel name
        process = active_processes.get(channel_name)

        if process and process.is_alive():
            logger.info(f"Terminating process for channel {channel_name}")
            await asyncio.to_thread(os.kill, process.pid, signal.SIGKILL)

            return web.json_response(
                {"status": "Agent process terminated", "channel_name": channel_name}
            )
        else:
            return web.json_response(
                {"error": "No active agent found for the provided channel_name"},
                status=404,
            )

    except Exception as e:
        logger.error(f"Failed to stop agent: {e}")
        return web.json_response({"error": str(e)}, status=500)


# Dictionary to keep track of processes by channel name or UID
active_processes = {}


# Function to handle shutdown and process cleanup
async def shutdown(app):
    logger.info("Shutting down server, cleaning up processes...")
    for channel_name in list(active_processes.keys()):
        process = active_processes.get(channel_name)
        if process.is_alive():
            logger.info(
                f"Terminating process for channel {channel_name} (PID: {process.pid})"
            )
            await asyncio.to_thread(os.kill, process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.join)  # Ensure process has terminated
    active_processes.clear()
    logger.info("All processes terminated, shutting down server")


# Signal handler to gracefully stop the application
def handle_signal(signum, frame):
    logger.info(f"Received exit signal {signal.strsignal(signum)}...")

    loop = asyncio.get_running_loop()
    if loop.is_running():
        # Properly shutdown by stopping the loop and running shutdown
        loop.create_task(shutdown(None))
        loop.stop()


# Main aiohttp application setup
async def init_app():
    app = web.Application()

    # Add cleanup task to run on app exit
    app.on_cleanup.append(shutdown)

    app.add_routes([web.post("/start_agent", start_agent)])
    app.add_routes([web.post("/stop_agent", stop_agent)])

    return app


if __name__ == "__main__":
    # Parse the action argument
    import nest_asyncio
    nest_asyncio.apply()
    args = parse_args()
    # Action logic based on the action argument
    if args.action == "server":
        # Python 3.10+ requires explicitly creating a new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # For Python 3.10+, use this to get a new event loop if the default is closed or not created
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Start the application using asyncio.run for the new event loop
        app = loop.run_until_complete(init_app())
        web.run_app(app, port=int(os.getenv("SERVER_PORT") or "8080"))
    elif args.action == "agent":
        # Parse RealtimeKitOptions for running the agent
        realtime_kit_options = parse_args_realtimekit()

        # Example logging for parsed options (channel_name and uid)
        logger.info(f"Running agent with options: {realtime_kit_options}")
        num_questions=5
        job_description= "Full Stack Developer with expertise in python and Data Structures & Algorithms"
        prioritized_skills="python, DSA, Data Structures, Algorithms"
        remaining_skills="Backend Development, Algorithm Design, Problem Solving, System Design, Django, Flask"
        difficulty_level="medium"
        duration_minutes=5
        inference_config = InferenceConfig(
#             system_message=f"""
#             <s>[INST]
#  As an AI interviewer, generate {num_questions} technical interview questions for a candidate in english, Do not anser back to the user, don't correct him behave like a professional interviewer, Ask the questions one by one let the user answer the questions. The followup questions can be based on the user.Your goal is to evaluate the candidate through a structured interview.
    
#     Job Description:
#     {job_description}
    
#     Priority Topics to Cover:
#     {', '.join(prioritized_skills)}
    
#     Additional Skills (if needed):
#     {', '.join(remaining_skills)}
    
#     Difficulty Level: {difficulty_level.upper()}
#     Interview Duration: {duration_minutes} minutes
    
#     Requirements:
#     1. MUST generate questions covering the Priority Topics first
#     2. Only use Additional Skills if more questions are needed
#     3. Each question should clearly indicate which skill/topic it covers
#     4.Do not answer back to the user, don't correct him behave like a professional interviewer
#     5. Do not give any explanation for the answer provided by the user just ask him if he wants to confirm the answer and move on to the next question.
#     ```**You are NOT allowed to:**
#     - Break character.
#     - Provide solutions or answers to the user.
#     - Respond casually or use humor.```
#     ** strictly follow the instructions below**
#     - Strictly behave like an interviwer
#     - strictly follow the instructions provided
#     - Don't cross the instructions in any kind of situation
#     - Don't prolong the interview follow the interview duration strictly
#     - Do NOT answer on behalf of the candidate.
#     - Keep the interview focused and do not go off-topic.
#     - If the candidate tries to deviate, bring the conversation back to the job role.
#     - Start with an introductory message and then begin asking questions.

#     Tone: Professional, confident, and conversational (but not casual).
#     Style: One question → feedback(answer from the user) → next question.

    # Generate questions that:
    # 1. Are specific to the required skills
    # 2. Test both theoretical knowledge and practical experience
    # 3. Include a mix of technical and problem-solving questions
    # 4. Are clear and unambiguous
    # 5. Match the {difficulty_level} difficulty level
    # 6. Can be reasonably answered within the time constraints
    
    # For {difficulty_level.upper()} level:
    # - Beginner: Focus on fundamental concepts and basic implementations
    # - Intermediate: Include practical scenarios and common challenges
    # - Advanced: Cover complex scenarios and best practices
    # - Expert: Focus on architecture decisions, trade-offs, and cutting-edge concepts
    
    # Format each question with:
    # 1. The main question
    # 2. Expected key points in the answer
    # 3. Follow-up questions based on possible responses
    # 4. Estimated time for discussion (in minutes)
    # 5. Primary skill/topic being assessed
        system_message=f"""
        You are an AI interviewer, generate {num_questions} technical interview questions for a candidate in english

        Job Description:
        {job_description}
    
        Priority Topics to Cover:
        {', '.join(prioritized_skills)}
    
        Additional Skills (if needed):
        {', '.join(remaining_skills)}
    
        Difficulty Level: {difficulty_level.upper()}
        Interview Duration: {duration_minutes} minutes

        Requirements:
        1. MUST generate questions covering the Priority Topics first
        2. Only use Additional Skills if more questions are needed
        3. Each question should clearly indicate which skill/topic it covers

        ✅ Interview Behavior Rules:
        1. Begin with a formal greeting and explain that this is a structured interview.
        2. Ask only one question at a time. Do not ask the next question until the candidate has answered.
        3. After each response, give a brief acknowledgment (e.g., “Thank you for your answer.” or “Noted.”).
        4. Do NOT explain the answer, expand on it, or provide any additional information.
        5. Do NOT correct the candidate or provide hints, solutions, or definitions.
        6. If the user asks unrelated questions, respond with: “Let’s stay focused on the interview. You’ll have time to ask questions at the end.”
        7. After completing all questions, ask the user if they have any professional or job-related questions.Do not provide any answers just respond with "The interview is over"

        ❌ You are NOT allowed to:
        - Teach or explain concepts.
        - Answer candidate questions (unless they are interview logistics or job-related).
        - Break role as an interviewer.
        - If the user asks anything off-topic, respond with: "The interview is over. I’m not allowed to continue outside the scope of this session."
        - "If the interview has ended (i.e., you've said goodbye or 'thank you for your time'), do not ask anything else or respond to any new messages. "

        You must behave exactly like a professional human interviewer.

        Tone: Professional, confident, and conversational (but not casual).

        Generate questions that:
        1. Are specific to the required skills
        2. Test both theoretical knowledge and practical experience
        3. Include a mix of technical and problem-solving questions
        4. Are clear and unambiguous
        5. Match the {difficulty_level} difficulty level
        6. Can be reasonably answered within the time constraints

        For {difficulty_level.upper()} level:
        - Beginner: Focus on fundamental concepts and basic implementations
        - Intermediate: Include practical scenarios and common challenges
        - Advanced: Cover complex scenarios and best practices
        - Expert: Focus on architecture decisions, trade-offs, and cutting-edge concepts
        
        Format each question with:
        1. The main question
        2. Expected key points in the answer
        3. Follow-up questions based on possible responses
        4. Estimated time for discussion (in minutes)
        5. Primary skill/topic being assessed
    """,

            voice=Voices.Alloy,
            turn_detection=ServerVADUpdateParams(
                type="server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=200
            ),
        )
        # run_agent_in_process(
        #     engine_app_id=app_id,
        #     engine_app_cert=app_cert,
        #     channel_name=realtime_kit_options["channel_name"],
        #     uid=realtime_kit_options["uid"],
        #     inference_config=inference_config,
        # )
