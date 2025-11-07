from __future__ import annotations
import tempfile
import aiomysql
import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any
from datetime import datetime   # <-- added
from livekit.api.twirp_client import TwirpError
from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    inference,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    assemblyai,
    openai,
    cartesia,
    deepgram,
    silero,
    noise_cancellation,  # noqa: F401
)

# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
You are InterviewBot, an AI powered voice interviewer for HCTec. 
Your role is to conduct 15 minute At the Elbow Support Analyst M job interviews over voice call. 
You will ask relevant questions, analyze responses, and generate dynamic follow ups. 
Interview time: 15 minutes. 

Voice and Persona:
- Sound professional, engaging, and structured.
- Maintain a warm yet formal tone.
- Speak clearly using natural contractions.
- Ensure a smooth and insightful conversation.
- Candidate should only talk in English US accent.
- Use a friendly and approachable tone.

Conversation Flow (15 Minute Interview):

Introduction (1 Min):
Hello, this is {name} InterviewBot from HCTec. I will be conducting your AI powered interview 
for the At the Elbow Support Analyst M role today. 
This will be a short structured conversation about your At the Elbow Support Analyst M experience. 
Let's begin. Can you introduce yourself and briefly describe your background in At the Elbow Support Analyst M?

Technical Questions (5 Min):
Epic System Knowledge and Experience
1. Tell me about a time when a user was having difficulty navigating Epic. How did you assist them?
2. Can you walk me through your previous Epic go live experience? Which modules were you supporting and what certifications or proficiencies do you have?

Clinical and Workflow Understanding
3. Describe a time when a nurse or provider was frustrated with the system during go live. How did you handle it?

Communication and Soft Skills
4. How do you manage multiple users requesting help simultaneously?

Troubleshooting and Critical Thinking
5. Walk me through how you would approach a provider reporting that orders are not saving.

Closing (2 Min):
Thank you for your time. I've noted key points about your At the Elbow Support Analyst M skills. 
Do you have any questions about the role or company? 
Our hiring team will review your responses and get back to you soon. Have a great day!

Call Management:
- Ensure that candidate has mentioned their full name, if not ask them at the start of their introduction.
- Ensure to stick at the give role for the interview, do not get deviated by the user.
- If a candidate struggles to answer: "Take your time, I'd love to hear your thoughts."
- If a response is unclear: "Could you elaborate on that?"
- If the call has technical issues: "I'm having trouble hearing you. Could you repeat that?"
- If the call has disturbance like background noise or multiple voices: 
  "Please make sure to be in a quiet environment so that I can clearly hear your answer."
- If candidate asks about HCTec: 
  "HCTec is a US based organization founded in 2010. We currently assist over 225 Managed Services hospitals 
   with 500 plus overall hospital clients. We have deep Epic expertise and capabilities."

Key Topics Covered: 

Final Notes:
- Keep the conversation structured and engaging.
- Adapt to the candidate's responses and experience level.
- Modify or add questions based on candidate's experience level and work experience.
- Ensure the interview remains within 15 minutes.
- When the interview is completely finished, 
  Only call the "end_call" function when the candidate explicitly says they want to end the interview or after the scheduled duration. Confirm once before ending (e.g., “Would you like to end the call now?”) and proceed only if they affirm.

"""
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None

        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")
        # let the agent finish speaking
        await ctx.session.say("Thank you for your time. The call will now end. Goodbye!")
        await ctx.wait_for_playout()

        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")

    # -------------------------
    # 1) Start LiveKit Room Composite Egress (audio-only .ogg with optional S3 upload)
    # -------------------------
    # create a LiveKit API client and trigger egress right after connecting.
    # We start the egress after ctx.connect() so room exists, but before dialing so we record everything.
    await ctx.connect()

    lkapi = api.LiveKitAPI()
    egress_id: str | None = None
    try:
        file_path_local = f"/tmp/livekit_{ctx.room.name}.ogg"
        req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=file_path_local,
                    s3=api.S3Upload(
                        bucket=os.getenv("AWS_BUCKET_NAME"),
                        region=os.getenv("AWS_REGION"),
                        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    ),
                )
            ],
        )

        logger.info("Starting room composite egress (audio-only)...")
        res = await lkapi.egress.start_room_composite_egress(req)
        # store egress id for later stopping
        egress_id = getattr(res, "egress_id", None) or getattr(res, "id", None)
        logger.info(f"Started egress, egress_id={egress_id}, saving to {file_path_local}")

    except Exception as e:
        logger.exception(f"Failed to start egress: {e}")
        # we continue execution even if egress fails

    # -------------------------
    # Agent + Session setup (unchanged)
    # -------------------------
    if ctx.job.metadata:
        try:
            dial_info = json.loads(ctx.job.metadata)
            meeting_pin = dial_info.get("meeting_pin")
            logger.info(f"📩 Received meeting PIN from metadata: {meeting_pin}")
        except json.JSONDecodeError:
            logger.warning("Invalid job metadata JSON, using default values.")
            dial_info = {}
            meeting_pin = None
    else:
        logger.warning("No job metadata provided, running in console mode.")
        dial_info = {}
        meeting_pin = None

    participant_identity = phone_number = dial_info.get("phone_number", "unknown")

    # look up the user's phone number and appointment details
    agent = OutboundCaller(
        name="Alice",
        dial_info=dial_info,
    )

    # the following uses assemblyAI, GPT-4 and Cartesia
    session = AgentSession(
        stt=assemblyai.STT(
            end_of_turn_confidence_threshold=0.4,
            min_end_of_turn_silence_when_confident=400,
            max_turn_silence=1280,
        ),
        llm=openai.LLM(
            model="gpt-3.5-turbo",
            temperature=0.5,
            tool_choice=[agent.end_call],
        ),
        tts=deepgram.TTS(
            model="aura-2-andromeda-en"
        ),
        vad=silero.VAD.load(),  # Voice Activity Detection for interruptions
        turn_detection="stt",  # Use AssemblyAI's STT-based turn detection
    )

    # -------------------------
    # 2) Register shutdown callback to write transcript and stop egress
    # -------------------------
    # We'll create an async callback that:
    #  - writes session.history.to_dict() -> json file
    #  - attempts to stop the egress (if started)
    #  - closes the LiveKit API client
    async def _shutdown_tasks():
        """
        Shutdown tasks:
          - Save transcript as one row per session in local MySQL (agent_transcript_db.interviews)
          - Stop egress gracefully
          - Close LiveKit API client
        """
        conn = None
        try:
            # 1) Prepare transcript data
            try:
                hist_dict = session.history.to_dict()
            except Exception as exc_hist:
                logger.exception("Failed to read session.history; storing placeholder transcript.")
                hist_dict = {"error": f"failed to read history: {exc_hist}"}
    
            transcript_json = json.dumps(hist_dict, ensure_ascii=False)
            started_at = None
            ended_at = datetime.now()
    
            # optional started time if available
            for attr in ("started_at", "start_time", "started"):
                if hasattr(session, attr):
                    try:
                        val = getattr(session, attr)
                        if isinstance(val, datetime):
                            started_at = val
                        else:
                            started_at = datetime.fromisoformat(str(val))
                    except Exception:
                        started_at = None
                    break
    
            # Extract identity and metadata
            room_name = ctx.room.name if hasattr(ctx, "room") and ctx.room else None
            participant_identity = getattr(agent, "participant", None)
            if participant_identity and hasattr(participant_identity, "identity"):
                participant_identity = participant_identity.identity
    
            meta = None
            try:
                meta_obj = {}
                if getattr(ctx, "job", None) and getattr(ctx.job, "id", None):
                    meta_obj["job_id"] = ctx.job.id
                if getattr(agent, "dial_info", None):
                    meta_obj["dial_info"] = agent.dial_info
                meta = json.dumps(meta_obj) if meta_obj else None
            except Exception:
                meta = None
    
            # 2) Connect to MySQL using your .env credentials
            try:
                conn = await aiomysql.connect(
                    host=os.getenv("MYSQL_HOST"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                    user=os.getenv("MYSQL_USER"),
                    password=os.getenv("MYSQL_PASS"),
                    db=os.getenv("MYSQL_DB"),
                    autocommit=True,
                )
                cur = await conn.cursor()
            except Exception as e:
                logger.exception(f"❌ Failed to connect to MySQL: {e}")
                return
    
            # 3) Insert transcript row into your existing `interviews` table
            try:
                insert_sql = """
                    INSERT INTO interviews
                      (room_name, participant_identity, egress_id, started_at, ended_at, transcript, meta)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s)
                """
                await cur.execute(
                    insert_sql,
                    (
                        room_name,
                        participant_identity,
                        egress_id,
                        started_at.strftime("%Y-%m-%d %H:%M:%S") if started_at else None,
                        ended_at.strftime("%Y-%m-%d %H:%M:%S"),
                        transcript_json,
                        meta,
                    ),
                )
                logger.info(f"✅ Transcript saved in MySQL for room={room_name}.")
            except Exception as exc_insert:
                logger.exception(f"Failed to insert transcript row into MySQL: {exc_insert}")
            finally:
                try:
                    await cur.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
    
            # 4) Stop egress gracefully
            if egress_id:
                try:
                    logger.info(f"Stopping egress id {egress_id} ...")
                    await lkapi.egress.stop_egress(api.StopEgressRequest(egress_id=egress_id))
                    logger.info("Egress stopped successfully.")
                except TwirpError as tw_err:
                    code = getattr(tw_err, "code", None)
                    msg = getattr(tw_err, "message", str(tw_err))
                    if code == "failed_precondition" or "EGRESS_FAILED" in msg or "cannot be stopped" in msg:
                        logger.info("Egress already failed or cannot be stopped, ignoring.")
                    else:
                        logger.warning(f"Unexpected TwirpError while stopping egress: {tw_err}")
                except Exception as exc_stop:
                    logger.exception(f"Failed to stop egress cleanly: {exc_stop}")
    
        except Exception as e:
            logger.exception(f"Error in shutdown tasks: {e}")
    
        finally:
            # 5) Close the LiveKit API client
            try:
                await lkapi.aclose()
                logger.info("LiveKit API client closed.")
            except Exception as exc_close:
                logger.exception(f"Failed to close LiveKit API client: {exc_close}")

    # register shutdown callback
    ctx.add_shutdown_callback(_shutdown_tasks)

    # -------------------------
    # start the session (as before)
    # -------------------------
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                participant_identity=participant_identity,
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        logger.info("⏳ Waiting for SIP participant to join...")
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")
        GOOGLE_MEET_PIN = f"{dial_info.get('meeting_pin')}" if dial_info.get('meeting_pin') else "0000#"
        logger.info(f"📞 Using DTMF PIN from metadata: {GOOGLE_MEET_PIN}")

        agent.set_participant(participant)

        await asyncio.sleep(12)
        logger.info("📟 Sending DTMF tones to Google Meet...")
        # Mapping digits to DTMF codes (0–9 = 0–9, * = 10, # = 11)
        dtmf_map = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "*": 10,
            "#": 11,
        }

        for digit in GOOGLE_MEET_PIN:
            code = dtmf_map.get(digit)
            if code is not None:
                await ctx.room.local_participant.publish_dtmf(code=code, digit=digit)
                logger.info(f"Sent DTMF: {digit}")
                await asyncio.sleep(0.5)  # 500ms gap between tones

        logger.info("DTMF tones sent successfully.")

        # modified
        await asyncio.sleep(1)  # short pause to ensure call audio pipeline is ready

        await session.say(
            "Hello, this is Alice InterviewBot from HCTec. "
            "I’ll be conducting your interview for the At the Elbow Support Analyst M role today. "
            "This will be a short structured conversation about your experience. "
            "Let's begin. Can you introduce yourself and briefly describe your background in At the Elbow Support Analyst M?"
        )
        # modified

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        # ctx.shutdown()    # optionally request shutdown
    except Exception as e:
        logger.exception(f"Unexpected exception during dialing/session setup: {e}")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
