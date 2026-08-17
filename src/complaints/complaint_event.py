# src/complaints/complaint_event.py

"""
Complaint Event -- REAL-TIME ENTRY

Two intake channels:

1. EMAIL
   - Connects to Gmail using IMAP
   - Fetches emails from the last 2 minutes
   - Extracts:
       Customer ID
       Phone number
       Sender email
       Subject
       Complaint text
       Complaint keywords

2. AUDIO
   - Takes an audio file
   - Uses Gemini to transcribe the audio
   - Extracts:
       Customer ID
       Phone number
       Complaint keywords

This module only extracts a clean ComplaintEvent.
It does not perform customer lookup, churn prediction,
offer generation, or email sending.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import imaplib
import email

from email.header import decode_header
from email.utils import parsedate_to_datetime

from dataclasses import dataclass, field

from datetime import datetime, timedelta

from typing import Optional, List

from dotenv import load_dotenv

from src.config import get, PROJECT_ROOT


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()


# ============================================================
# COMPLAINT KEYWORDS
# ============================================================

DEFAULT_KEYWORDS = [

    "cancel",
    "cancel my",
    "cancel service",
    "close my account",

    "switch provider",
    "switch to",
    "moving to another",

    "disconnect",
    "terminate",
    "leaving",
    "leave your service",

    "not happy",
    "unhappy",
    "dissatisfied",
    "frustrated",
    "unacceptable",

    "complaint",
    "complain",

    "refund",
    "compensation",

    "poor service",
    "bad service",
    "overcharged",
    "billing issue",

    "competitor",
    "better deal",
    "cheaper plan",
    "downgrade",

    "escalate",
    "speak to a manager",
    "speak to manager",
]


COMPLAINT_KEYWORDS = get(
    "complaints",
    "keywords",
    default=DEFAULT_KEYWORDS
)


# ============================================================
# CUSTOMER ID PATTERN
#
# Example:
# 0404-SWRVG
# ============================================================

CUSTOMER_ID_PATTERN = re.compile(
    r"\b\d{4}-[A-Z]{5}\b",
    re.IGNORECASE
)


# ============================================================
# PHONE NUMBER PATTERN
# ============================================================

PHONE_PATTERN = re.compile(
    r"(\+?\d[\d\-.\s()]{7,16}\d)"
)


# ============================================================
# COMPLAINT EVENT
# ============================================================

@dataclass
class ComplaintEvent:

    source: str

    received_at: datetime

    raw_text: str

    customer_id: Optional[str] = None

    contact_number: Optional[str] = None

    sender_email: Optional[str] = None

    subject: Optional[str] = None

    matched_keywords: List[str] = field(
        default_factory=list
    )

    @property
    def missing_customer_info(self) -> bool:

        return (
            not self.customer_id
            and not self.contact_number
        )

    @property
    def is_complaint(self) -> bool:

        # Every email received in support inbox
        # is considered a complaint/contact.

        if self.source == "email":
            return True

        # Audio is considered a complaint only
        # when complaint keywords are found.

        return len(
            self.matched_keywords
        ) > 0


# ============================================================
# EXTRACT CUSTOMER ID
# ============================================================

def extract_customer_id(
    text: str
) -> Optional[str]:

    match = CUSTOMER_ID_PATTERN.search(
        text or ""
    )

    if match:

        return match.group(0).upper()

    return None


# ============================================================
# EXTRACT PHONE NUMBER
# ============================================================

def extract_contact_number(
    text: str
) -> Optional[str]:

    match = PHONE_PATTERN.search(
        text or ""
    )

    if not match:

        return None

    digits_only = re.sub(
        r"\D",
        "",
        match.group(1)
    )

    if len(digits_only) < 8:

        return None

    return match.group(1).strip()


# ============================================================
# FIND COMPLAINT KEYWORDS
# ============================================================

def find_matched_keywords(
    text: str
) -> List[str]:

    text_lower = (
        text or ""
    ).lower()

    return [

        keyword

        for keyword in COMPLAINT_KEYWORDS

        if keyword.lower() in text_lower

    ]


# ============================================================
# DECODE EMAIL HEADER
# ============================================================

def _decode_mime_header(
    value: str
) -> str:

    if not value:

        return ""

    decoded = ""

    for text, encoding in decode_header(value):

        if isinstance(text, bytes):

            decoded += text.decode(
                encoding or "utf-8",
                errors="replace"
            )

        else:

            decoded += text

    return decoded


# ============================================================
# GET EMAIL BODY
# ============================================================

def _get_email_body(
    msg: "email.message.Message"
) -> str:

    """
    Extract the plain-text body.

    Priority:
        1. text/plain
        2. fallback readable part
    """

    # --------------------------------------------------------
    # Multipart email
    # --------------------------------------------------------

    if msg.is_multipart():

        # First look for text/plain
        for part in msg.walk():

            content_type = (
                part.get_content_type()
            )

            disposition = str(
                part.get(
                    "Content-Disposition",
                    ""
                )
            )

            if (
                content_type == "text/plain"
                and "attachment" not in disposition
            ):

                payload = part.get_payload(
                    decode=True
                )

                if payload:

                    return payload.decode(
                        part.get_content_charset()
                        or "utf-8",
                        errors="replace"
                    )


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        for part in msg.walk():

            payload = part.get_payload(
                decode=True
            )

            if payload:

                return payload.decode(
                    part.get_content_charset()
                    or "utf-8",
                    errors="replace"
                )

        return ""


    # --------------------------------------------------------
    # Normal non-multipart email
    # --------------------------------------------------------

    payload = msg.get_payload(
        decode=True
    )

    if payload:

        return payload.decode(
            msg.get_content_charset()
            or "utf-8",
            errors="replace"
        )

    return str(
        msg.get_payload()
    )


# ============================================================
# EXTRACT COMPLAINT FROM ONE EMAIL
# ============================================================

def extract_complaint_from_email(
    msg: "email.message.Message"
) -> ComplaintEvent:

    # --------------------------------------------------------
    # Sender
    # --------------------------------------------------------

    sender_raw = _decode_mime_header(
        msg.get(
            "From",
            ""
        )
    )

    sender_match = re.search(
        r"[\w.\-+]+@[\w.\-]+",
        sender_raw
    )

    if sender_match:

        sender_email = (
            sender_match.group(0)
        )

    else:

        sender_email = sender_raw


    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subject = _decode_mime_header(
        msg.get(
            "Subject",
            ""
        )
    )


    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    body = _get_email_body(
        msg
    )


    # --------------------------------------------------------
    # Combine subject + body
    # --------------------------------------------------------

    full_text = (
        f"{subject}\n{body}"
    )


    # --------------------------------------------------------
    # Create ComplaintEvent
    # --------------------------------------------------------

    return ComplaintEvent(

        source="email",

        received_at=datetime.now().astimezone(),

        raw_text=body.strip(),

        customer_id=extract_customer_id(
            full_text
        ),

        contact_number=extract_contact_number(
            full_text
        ),

        sender_email=sender_email,

        subject=subject,

        matched_keywords=find_matched_keywords(
            full_text
        ),
    )


# ============================================================
# CONVERT EMAIL TIME TO LOCAL TIME
# ============================================================

def _convert_to_local_time(
    dt: datetime
) -> datetime:

    """
    Convert email Date header to local timezone.

    If the email Date header has no timezone,
    use the computer's local timezone.
    """

    local_now = datetime.now().astimezone()

    if dt.tzinfo is None:

        return dt.replace(
            tzinfo=local_now.tzinfo
        )

    return dt.astimezone(
        local_now.tzinfo
    )


# ============================================================
# GET EMAIL DATE
# ============================================================

def _get_email_datetime(
    msg: "email.message.Message"
) -> Optional[datetime]:

    """
    Get the Date header from an email and
    convert it to local timezone.

    Example:

    Date:
    Mon, 17 Aug 2026 11:37:20 +0530
    """

    date_header = msg.get(
        "Date"
    )

    if not date_header:

        return None

    try:

        email_time = (
            parsedate_to_datetime(
                date_header
            )
        )

        return _convert_to_local_time(
            email_time
        )

    except Exception as e:

        print(
            "[complaint_event] "
            f"Date parsing error: {e}"
        )

        return None


# ============================================================
# FETCH EMAILS FROM LAST 2 MINUTES
# ============================================================

def fetch_new_complaint_emails() -> List[ComplaintEvent]:

    """
    Fetch emails from the last 2 minutes.

    IMPORTANT:

    We do NOT use:

        INTERNALDATE
        UID checkpoint
        UNSEEN

    Instead:

        Gmail search
              ↓
        Get complete email
              ↓
        Read Date header
              ↓
        Compare with current time
              ↓
        Accept if inside 2-minute window
    """

    # ========================================================
    # READ IMAP SETTINGS
    # ========================================================

    host = os.getenv(
        "IMAP_HOST"
    )

    port = int(
        os.getenv(
            "IMAP_PORT",
            "993"
        )
    )

    user = os.getenv(
        "IMAP_USER"
    )

    password = os.getenv(
        "IMAP_PASSWORD"
    )


    # ========================================================
    # CHECK SETTINGS
    # ========================================================

    if not all([
        host,
        user,
        password
    ]):

        raise EnvironmentError(

            "IMAP_HOST, IMAP_USER, "
            "and IMAP_PASSWORD must be "
            "set in .env"

        )


    # ========================================================
    # CURRENT TIME
    # ========================================================

    now = datetime.now().astimezone()


    # ========================================================
    # LAST 2 MINUTES
    # ========================================================

    two_minutes_ago = (
        now - timedelta(
            minutes=2
        )
    )


    # ========================================================
    # DISPLAY TIME WINDOW
    # ========================================================

    print("=" * 60)

    print(
        "[complaint_event] EMAIL FETCH"
    )

    print("=" * 60)

    print(
        "[complaint_event] "
        f"Current time: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )

    print(
        "[complaint_event] "
        f"Checking emails after: "
        f"{two_minutes_ago.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )


    events: List[ComplaintEvent] = []


    # ========================================================
    # CONNECT TO GMAIL
    # ========================================================

    try:

        with imaplib.IMAP4_SSL(
            host,
            port
        ) as imap:


            # =================================================
            # LOGIN
            # =================================================

            login_status, login_response = (
                imap.login(
                    user,
                    password
                )
            )

            print(
                "[complaint_event] "
                f"Login: {login_status}"
            )


            if login_status != "OK":

                print(
                    "[complaint_event] "
                    "Gmail login failed."
                )

                return events


            # =================================================
            # SELECT INBOX
            # =================================================

            select_status, select_response = (
                imap.select(
                    "INBOX"
                )
            )

            print(
                "[complaint_event] "
                f"Selected INBOX: "
                f"{select_status}"
            )


            if select_status != "OK":

                print(
                    "[complaint_event] "
                    "Could not select INBOX."
                )

                return events


            # =================================================
            # SEARCH EMAILS
            # =================================================
            #
            # IMAP SINCE works using dates, not minutes.
            #
            # Therefore search from yesterday.
            #
            # We perform the exact 2-minute filtering
            # ourselves below.
            # =================================================

            yesterday = (
                now - timedelta(
                    days=1
                )
            )

            search_date = (
                yesterday.strftime(
                    "%d-%b-%Y"
                )
            )


            status, uid_data = (
                imap.uid(
                    "search",
                    None,
                    "SINCE",
                    search_date
                )
            )


            if status != "OK":

                print(
                    "[complaint_event] "
                    "IMAP search failed."
                )

                return events


            # =================================================
            # GET UID LIST
            # =================================================

            candidate_uids = (

                uid_data[0].split()

                if (
                    uid_data
                    and uid_data[0]
                )

                else []

            )


            print(
                "[complaint_event] "
                f"Candidate emails: "
                f"{len(candidate_uids)}"
            )


            # =================================================
            # CHECK EACH EMAIL
            # =================================================

            for uid in candidate_uids:

                print("-" * 60)

                print(
                    "[complaint_event] "
                    f"Checking UID: {uid}"
                )


                # =================================================
                # FETCH COMPLETE EMAIL
                # =================================================

                status, msg_data = (
                    imap.uid(
                        "fetch",
                        uid,
                        "(RFC822)"
                    )
                )


                if status != "OK":

                    print(
                        "[complaint_event] "
                        f"Could not fetch UID "
                        f"{uid}. Skipping."
                    )

                    continue


                # =================================================
                # EXTRACT RAW EMAIL
                # =================================================

                raw_email = None


                for item in msg_data:

                    if not isinstance(
                        item,
                        tuple
                    ):

                        continue


                    if len(item) < 2:

                        continue


                    if isinstance(
                        item[1],
                        bytes
                    ):

                        raw_email = item[1]

                        break


                if raw_email is None:

                    print(
                        "[complaint_event] "
                        f"Could not extract raw "
                        f"email for UID {uid}. "
                        "Skipping."
                    )

                    continue


                # =================================================
                # CONVERT TO EMAIL MESSAGE
                # =================================================

                try:

                    msg = (
                        email.message_from_bytes(
                            raw_email
                        )
                    )

                except Exception as e:

                    print(
                        "[complaint_event] "
                        f"Could not parse UID "
                        f"{uid}: {e}"
                    )

                    continue


                # =================================================
                # GET EMAIL TIME
                # =================================================

                email_time = (
                    _get_email_datetime(
                        msg
                    )
                )


                if email_time is None:

                    print(
                        "[complaint_event] "
                        f"UID {uid} has no "
                        "valid Date header. "
                        "Skipping."
                    )

                    continue


                # =================================================
                # SHOW EMAIL TIME
                # =================================================

                print(
                    "[complaint_event] "
                    f"Email time: "
                    f"{email_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )


                # =================================================
                # CHECK WHETHER EMAIL IS TOO OLD
                # =================================================

                if email_time < two_minutes_ago:

                    print(
                        "[complaint_event] "
                        f"UID {uid} is older "
                        "than 2 minutes. "
                        "Skipping."
                    )

                    continue


                # =================================================
                # CHECK FUTURE DATE
                # =================================================

                if email_time > now:

                    print(
                        "[complaint_event] "
                        f"UID {uid} has a "
                        "future timestamp. "
                        "Skipping."
                    )

                    continue


                # =================================================
                # EXTRACT COMPLAINT
                # =================================================

                event = (
                    extract_complaint_from_email(
                        msg
                    )
                )


                # Use actual email time
                event.received_at = (
                    email_time
                )


                # =================================================
                # ADD EVENT
                # =================================================

                events.append(
                    event
                )


                # =================================================
                # DISPLAY EVENT
                # =================================================

                print(
                    "[complaint_event] "
                    f"UID {uid} ACCEPTED!"
                )

                print(
                    f"    From       : "
                    f"{event.sender_email}"
                )

                print(
                    f"    Subject    : "
                    f"{event.subject}"
                )

                print(
                    f"    Customer ID: "
                    f"{event.customer_id}"
                )

                print(
                    f"    Phone      : "
                    f"{event.contact_number}"
                )

                print(
                    f"    Keywords   : "
                    f"{event.matched_keywords}"
                )


    # ========================================================
    # IMAP ERROR
    # ========================================================

    except imaplib.IMAP4.error as e:

        print(
            "[complaint_event] "
            f"IMAP ERROR: {e}"
        )

        return events


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print(
            "[complaint_event] "
            f"ERROR: {type(e).__name__}: {e}"
        )

        return events


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("=" * 60)

    print(
        "[complaint_event] "
        f"Found {len(events)} email(s) "
        "from the last 2 minutes."
    )

    print("=" * 60)


    return events


# ============================================================
# AUDIO CHANNEL
# ============================================================

def transcribe_audio(
    audio_path: str
) -> str:

    """
    Transcribe an audio file using Gemini.

    Uses the same Gemini client already
    configured in genai/llm_client.py.
    """

    from genai.llm_client import (
        client,
        MODEL_NAME
    )

    from google.genai import types


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )


    # ========================================================
    # READ AUDIO
    # ========================================================

    with open(
        audio_path,
        "rb"
    ) as f:

        audio_bytes = f.read()


    # ========================================================
    # FILE EXTENSION
    # ========================================================

    ext = (
        os.path.splitext(
            audio_path
        )[1]
        .lower()
        .lstrip(".")
    )


    # ========================================================
    # MIME TYPE
    # ========================================================

    mime_map = {

        "wav": "audio/wav",

        "mp3": "audio/mp3",

        "m4a": "audio/mp4",

        "ogg": "audio/ogg",

    }


    mime_type = mime_map.get(
        ext,
        "audio/wav"
    )


    # ========================================================
    # GEMINI TRANSCRIPTION
    # ========================================================

    response = (
        client.models.generate_content(

            model=MODEL_NAME,

            contents=[

                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ),

                (
                    "Transcribe this audio call "
                    "exactly as spoken, word for word. "
                    "Output ONLY the transcript text, "
                    "no commentary."
                ),

            ],
        )
    )


    return (
        response.text or ""
    ).strip()


# ============================================================
# EXTRACT COMPLAINT FROM AUDIO
# ============================================================

def extract_complaint_from_audio(
    audio_path: str
) -> ComplaintEvent:

    transcript = (
        transcribe_audio(
            audio_path
        )
    )


    return ComplaintEvent(

        source="audio",

        received_at=(
            datetime.now().astimezone()
        ),

        raw_text=transcript,

        customer_id=extract_customer_id(
            transcript
        ),

        contact_number=extract_contact_number(
            transcript
        ),

        sender_email=None,

        subject=os.path.basename(
            audio_path
        ),

        matched_keywords=find_matched_keywords(
            transcript
        ),

    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "COMPLAINT EVENT -- manual test"
    )

    print("=" * 60)


    # ========================================================
    # ASK CHANNEL
    # ========================================================

    choice = input(
        "Test which channel? [email/audio]: "
    ).strip().lower()


    # ========================================================
    # EMAIL
    # ========================================================

    if choice == "email":

        found = (
            fetch_new_complaint_emails()
        )


        print()

        print(
            f"Found {len(found)} "
            "email(s) from the last 2 minutes."
        )


        for event in found:

            print("-" * 60)

            print(
                f"From          : "
                f"{event.sender_email}"
            )

            print(
                f"Subject       : "
                f"{event.subject}"
            )

            print(
                f"Received At   : "
                f"{event.received_at}"
            )

            print(
                f"Customer ID   : "
                f"{event.customer_id}"
            )

            print(
                f"Contact #     : "
                f"{event.contact_number}"
            )

            print(
                f"Keywords hit  : "
                f"{event.matched_keywords}"
            )

            print(
                f"Missing info? : "
                f"{event.missing_customer_info}"
            )

            print(
                f"Is complaint? : "
                f"{event.is_complaint}"
            )

            print(
                f"Message       : "
                f"{event.raw_text[:500]}"
            )


    # ========================================================
    # AUDIO
    # ========================================================

    elif choice == "audio":

        path = input(
            "Path to audio file: "
        ).strip()


        try:

            event = (
                extract_complaint_from_audio(
                    path
                )
            )


            print("-" * 60)

            print(
                f"Transcript    : "
                f"{event.raw_text[:300]}..."
            )

            print(
                f"Customer ID   : "
                f"{event.customer_id}"
            )

            print(
                f"Contact #     : "
                f"{event.contact_number}"
            )

            print(
                f"Keywords hit  : "
                f"{event.matched_keywords}"
            )

            print(
                f"Is complaint? : "
                f"{event.is_complaint}"
            )

            print(
                f"Missing info? : "
                f"{event.missing_customer_info}"
            )


        except Exception as e:

            print(
                f"Audio error: {e}"
            )


    # ========================================================
    # INVALID OPTION
    # ========================================================

    else:

        print(
            "Enter 'email' or 'audio'."
        )