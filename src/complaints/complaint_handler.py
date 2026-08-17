# src/complaints/complaint_handler.py

"""
Complaint Handler

Responsibilities:
1. Poll Gmail continuously for new complaint emails.
2. Extract customer information.
3. Validate Customer ID.
4. Save valid complaints.
5. Run customer prediction/risk pipeline.
6. Generate retention offer.
7. Store offer as pending approval.
8. Allow human approval before sending an email.
9. Run automatically while FastAPI backend is running.

The polling thread automatically stops when the FastAPI process stops.
"""

import os
import json
import time
import smtplib
import threading

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config import get, PROJECT_ROOT
from src.data.loader import load_raw_data
from src.prediction.predictor import predict_customer
from src.complaints.complaint_event import ComplaintEvent


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

PENDING_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pending_complaint_replies.json"
)

MANUAL_REVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manual_review_queue.json"
)

COMPLAINTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "complaints.json"
)


# ============================================================
# POLLING CONTROL
# ============================================================

_polling_thread = None
_polling_stop_event = threading.Event()


# ============================================================
# JSON HELPERS
# ============================================================

def _load_json_store(path: Path) -> dict:

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:

        print(
            f"[complaint_handler] Failed to read "
            f"{path}: {exc}"
        )

        return {}


def _save_json_store(path: Path, data: dict) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )


# ============================================================
# SAVE COMPLAINT
# ============================================================

def _save_valid_email_complaint(
    complaint: ComplaintEvent,
) -> str:

    complaints = _load_json_store(
        COMPLAINTS_PATH
    )

    complaint_id = (
        datetime.now().strftime(
            "%Y%m%d%H%M%S%f"
        )
        + "-"
        + str(complaint.customer_id)
    )

    complaints[complaint_id] = {

        "complaint_id": complaint_id,

        "source": "email",

        "customer_id": complaint.customer_id,

        "contact_number": complaint.contact_number,

        "sender_email": complaint.sender_email,

        "subject": complaint.subject,

        "complaint_text": complaint.raw_text,

        "matched_keywords": (
            complaint.matched_keywords
        ),

        "received_at": (
            complaint.received_at.isoformat()
        ),

        "status": "new",
    }

    _save_json_store(
        COMPLAINTS_PATH,
        complaints,
    )

    print(
        f"[complaint_handler] "
        f"Complaint saved: {complaint_id}"
    )

    return complaint_id


# ============================================================
# CUSTOMER LOOKUP
# ============================================================

def find_customer_row(
    customer_id: Optional[str],
    contact_number: Optional[str],
) -> Optional[dict]:

    df = load_raw_data()

    df["CustomerID"] = (
        df["CustomerID"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------

    if customer_id:

        normalized_id = (
            str(customer_id)
            .strip()
            .upper()
        )

        match = df[
            df["CustomerID"]
            .str.upper()
            == normalized_id
        ]

        if not match.empty:

            return (
                match
                .iloc[0]
                .to_dict()
            )

    # --------------------------------------------------------
    # Contact number
    # --------------------------------------------------------
    # Dataset currently doesn't contain
    # an actual customer phone number.
    #
    # This remains here for future CRM integration.

    if contact_number:
        pass

    return None


# ============================================================
# EMAIL SENDING
# ============================================================

def _send_email(
    to_email: str,
    subject: str,
    body: str,
) -> None:

    host = os.getenv(
        "SMTP_HOST"
    )

    port = int(
        os.getenv(
            "SMTP_PORT",
            "587",
        )
    )

    user = os.getenv(
        "SMTP_USER"
    )

    password = os.getenv(
        "SMTP_PASSWORD"
    )

    from_email = os.getenv(
        "SMTP_FROM_EMAIL",
        user,
    )

    if not all(
        [
            host,
            user,
            password,
        ]
    ):

        raise EnvironmentError(
            "SMTP_HOST, SMTP_USER, and "
            "SMTP_PASSWORD must be set "
            "in .env"
        )

    msg = MIMEMultipart()

    msg["From"] = from_email

    msg["To"] = to_email

    msg["Subject"] = subject

    msg.attach(
        MIMEText(
            body,
            "plain",
        )
    )

    with smtplib.SMTP(
        host,
        port,
    ) as server:

        server.starttls()

        server.login(
            user,
            password,
        )

        server.sendmail(
            from_email,
            [to_email],
            msg.as_string(),
        )


def _reply_subject(
    original_subject: Optional[str],
) -> str:

    original_subject = (
        original_subject
        or "your message to us"
    )

    if original_subject.lower().startswith(
        "re:"
    ):
        return original_subject

    return f"Re: {original_subject}"


# ============================================================
# MISSING CUSTOMER INFORMATION
# ============================================================

def send_missing_info_request(
    complaint: ComplaintEvent,
) -> dict:

    body = (
        "Hi,\n\n"
        "Thanks for reaching out. To look into "
        "this for you, could you please reply "
        "with your Customer ID or the phone "
        "number registered on your account?\n\n"
        "Once we have that, our team will review "
        "your account and get back to you.\n\n"
        "Thanks,\n"
        "Customer Care Team"
    )

    _send_email(
        complaint.sender_email,
        _reply_subject(
            complaint.subject
        ),
        body,
    )

    return {
        "status": "info_requested",
        "sender_email": complaint.sender_email,
    }


# ============================================================
# CUSTOMER ID NOT FOUND
# ============================================================

def send_id_not_found_request(
    complaint: ComplaintEvent,
) -> dict:

    body = (
        "Hi,\n\n"
        "Thanks for reaching out. We couldn't "
        "find an account matching the details "
        "provided.\n\n"
        f"Customer ID received: "
        f"{complaint.customer_id or 'not recognized'}\n\n"
        "Could you please double-check your "
        "Customer ID and reply with the correct "
        "one? You can find it on your bill or "
        "account page.\n\n"
        "Thanks,\n"
        "Customer Care Team"
    )

    _send_email(
        complaint.sender_email,
        _reply_subject(
            complaint.subject
        ),
        body,
    )

    return {
        "status": "id_not_found",
        "sender_email": complaint.sender_email,
    }


# ============================================================
# DRAFT OFFER EMAIL
# ============================================================

def _draft_offer_reply_body(
    customer_id: str,
    predictor_result: dict,
) -> str:

    if (
        predictor_result.get(
            "agent_approved"
        )
        and predictor_result.get("offer")
    ):

        offer_text = (
            predictor_result["offer"]
        )

        return (
            "Hi,\n\n"
            "Thank you for getting in touch, "
            "and sorry to hear about your "
            "experience.\n\n"
            "We'd like to make things right. "
            "Based on your account, here's "
            "what we can offer:\n\n"
            f"{offer_text}\n\n"
            "If you'd like to go ahead with "
            "this, just reply to this email "
            "and we'll get it set up.\n\n"
            "Thanks,\n"
            "Customer Care Team"
        )

    return (
        "Hi,\n\n"
        "Thank you for getting in touch, "
        "and sorry to hear about your "
        "experience.\n\n"
        "Your case has been passed to a "
        "member of our retention team, who "
        "will review your account and reach "
        "out to you directly.\n\n"
        "Thanks,\n"
        "Customer Care Team"
    )


# ============================================================
# HANDLE EMAIL COMPLAINT
# ============================================================

def handle_email_complaint(
    complaint: ComplaintEvent,
) -> dict:

    print(
        "\n[complaint_handler] "
        "Processing complaint email..."
    )

    # --------------------------------------------------------
    # Missing customer information
    # --------------------------------------------------------

    if complaint.missing_customer_info:

        print(
            "[complaint_handler] "
            "Customer ID/contact number missing."
        )

        return send_missing_info_request(
            complaint
        )

    # --------------------------------------------------------
    # Find customer
    # --------------------------------------------------------

    customer_row = find_customer_row(
        complaint.customer_id,
        complaint.contact_number,
    )

    if customer_row is None:

        print(
            "[complaint_handler] "
            f"Customer not found: "
            f"{complaint.customer_id}"
        )

        return send_id_not_found_request(
            complaint
        )

    # --------------------------------------------------------
    # Save complaint
    # --------------------------------------------------------

    complaint_id = (
        _save_valid_email_complaint(
            complaint
        )
    )

    # --------------------------------------------------------
    # Run prediction pipeline
    # --------------------------------------------------------

    print(
        f"[complaint_handler] "
        f"Running prediction for "
        f"{complaint.customer_id}"
    )

    result = predict_customer(
        customer_row,
        run_agent=True,
    )

    # --------------------------------------------------------
    # Create draft
    # --------------------------------------------------------

    draft_body = _draft_offer_reply_body(
        complaint.customer_id,
        result,
    )

    # --------------------------------------------------------
    # Save pending approval
    # --------------------------------------------------------

    pending = _load_json_store(
        PENDING_STORE_PATH
    )

    pending[
        complaint.customer_id
    ] = {

        "customer_id":
            complaint.customer_id,

        "complaint_id":
            complaint_id,

        "sender_email":
            complaint.sender_email,

        "subject":
            _reply_subject(
                complaint.subject
            ),

        "draft_body":
            draft_body,

        "churn_probability":
            result.get(
                "churn_probability",
                0.0,
            ),

        "risk_tier":
            result.get(
                "risk_tier"
            ),

        "agent_approved":
            result.get(
                "agent_approved",
                False,
            ),

        "agent_escalated":
            result.get(
                "agent_escalated",
                False,
            ),

        "status":
            "pending_approval",

        "created_at":
            datetime.now().isoformat(),
    }

    _save_json_store(
        PENDING_STORE_PATH,
        pending,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "COMPLAINT PROCESSED"
    )

    print(
        "=" * 60
    )

    print(
        f"Customer ID       : "
        f"{complaint.customer_id}"
    )

    print(
        f"Complaint ID      : "
        f"{complaint_id}"
    )

    print(
        f"Risk tier         : "
        f"{result.get('risk_tier')}"
    )

    print(
        f"Churn probability : "
        f"{result.get('churn_probability', 0):.2%}"
    )

    print(
        "Status            : "
        "pending_approval"
    )

    print(
        "=" * 60
        + "\n"
    )

    return {

        "status":
            "pending_approval",

        "customer_id":
            complaint.customer_id,

        "complaint_id":
            complaint_id,

        "risk_tier":
            result.get(
                "risk_tier"
            ),

        "churn_probability":
            result.get(
                "churn_probability",
                0.0,
            ),
    }


# ============================================================
# APPROVE / REJECT REPLY
# ============================================================

def approve_and_send_reply(
    customer_id: str,
    approve: bool = True,
    approved_by: str = "ops",
) -> dict:

    pending = _load_json_store(
        PENDING_STORE_PATH
    )

    record = pending.get(
        customer_id
    )

    if record is None:

        print(
            "[complaint_handler] "
            f"No pending reply found "
            f"for {customer_id}"
        )

        return {
            "status": "not_found",
            "customer_id": customer_id,
        }

    # --------------------------------------------------------
    # Reject
    # --------------------------------------------------------

    if not approve:

        record["status"] = "rejected"

        record["decided_by"] = (
            approved_by
        )

        record["decided_at"] = (
            datetime.now().isoformat()
        )

        pending[customer_id] = record

        _save_json_store(
            PENDING_STORE_PATH,
            pending,
        )

        print(
            f"[complaint_handler] "
            f"Reply rejected for "
            f"{customer_id}"
        )

        return record

    # --------------------------------------------------------
    # Approve and send
    # --------------------------------------------------------

    _send_email(
        record["sender_email"],
        record["subject"],
        record["draft_body"],
    )

    record["status"] = (
        "approved_sent"
    )

    record["decided_by"] = (
        approved_by
    )

    record["decided_at"] = (
        datetime.now().isoformat()
    )

    pending[customer_id] = record

    _save_json_store(
        PENDING_STORE_PATH,
        pending,
    )

    print(
        f"[complaint_handler] "
        f"Reply APPROVED and SENT "
        f"to {record['sender_email']}"
    )

    return record


# ============================================================
# AUDIO COMPLAINT
# ============================================================

def handle_audio_complaint(
    complaint: ComplaintEvent,
) -> Optional[dict]:

    if not complaint.is_complaint:

        print(
            "[complaint_handler] "
            "No complaint detected."
        )

        return None

    # --------------------------------------------------------
    # Missing customer
    # --------------------------------------------------------

    if complaint.missing_customer_info:

        review_queue = _load_json_store(
            MANUAL_REVIEW_PATH
        )

        entry_id = (
            complaint.subject
            or datetime.now().isoformat()
        )

        review_queue[entry_id] = {

            "source": "audio",

            "transcript":
                complaint.raw_text,

            "matched_keywords":
                complaint.matched_keywords,

            "reason":
                "Customer could not be "
                "identified from audio.",

            "queued_at":
                datetime.now().isoformat(),
        }

        _save_json_store(
            MANUAL_REVIEW_PATH,
            review_queue,
        )

        return {
            "status":
                "manual_review_required"
        }

    # --------------------------------------------------------
    # Customer lookup
    # --------------------------------------------------------

    customer_row = find_customer_row(
        complaint.customer_id,
        complaint.contact_number,
    )

    if customer_row is None:

        review_queue = _load_json_store(
            MANUAL_REVIEW_PATH
        )

        entry_id = (
            complaint.customer_id
            or complaint.subject
        )

        review_queue[entry_id] = {

            "source": "audio",

            "transcript":
                complaint.raw_text,

            "customer_id_heard":
                complaint.customer_id,

            "reason":
                f"CustomerID "
                f"'{complaint.customer_id}' "
                f"not found.",

            "queued_at":
                datetime.now().isoformat(),
        }

        _save_json_store(
            MANUAL_REVIEW_PATH,
            review_queue,
        )

        return {
            "status":
                "manual_review_required"
        }

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict_customer(
        customer_row,
        run_agent=True,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "AUDIO COMPLAINT PROCESSED"
    )

    print(
        "=" * 60
    )

    print(
        f"Customer ID       : "
        f"{complaint.customer_id}"
    )

    print(
        f"Churn probability : "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Risk tier         : "
        f"{result['risk_tier']}"
    )

    print(
        "=" * 60
        + "\n"
    )

    return {

        "status":
            "processed",

        "customer_id":
            complaint.customer_id,

        "risk_tier":
            result["risk_tier"],

        "churn_probability":
            result["churn_probability"],
    }


# ============================================================
# SINGLE POLL
# ============================================================

def poll_complaint_emails_once() -> int:
    """
    Check Gmail once and process all new complaint emails.
    """

    from src.complaints.complaint_event import (
        fetch_new_complaint_emails
    )

    try:

        new_emails = (
            fetch_new_complaint_emails()
        )

    except Exception as exc:

        print(
            "[complaint_handler] "
            f"Gmail polling error: {exc}"
        )

        return 0

    if not new_emails:

        print(
            "[complaint_handler] "
            "No new complaint emails."
        )

        return 0

    print(
        "[complaint_handler] "
        f"Found {len(new_emails)} "
        f"new complaint email(s)."
    )

    processed = 0

    for complaint in new_emails:

        try:

            handle_email_complaint(
                complaint
            )

            processed += 1

        except Exception as exc:

            print(
                "[complaint_handler] "
                f"Failed to process complaint: "
                f"{exc}"
            )

    return processed


# ============================================================
# EMAIL POLLING LOOP
# ============================================================

def start_email_polling(
    run_once: bool = False,
) -> None:
    """
    Continuously poll Gmail.

    This function blocks when called directly.

    run_once=True:
        Perform one check and return.

    run_once=False:
        Continue until stop_email_polling()
        is called or the backend process exits.
    """

    interval = get(
        "complaints",
        "poll_interval_seconds",
        default=60,
    )

    try:
        interval = int(interval)
    except Exception:
        interval = 60

    print(
        "\n"
        + "=" * 60
    )

    print(
        "COMPLAINT EMAIL AUTO-POLLING STARTED"
    )

    print(
        f"Checking Gmail every {interval} seconds."
    )

    print(
        "Polling will run while the backend is running."
    )

    print(
        "=" * 60
        + "\n"
    )

    while not _polling_stop_event.is_set():

        poll_complaint_emails_once()

        if run_once:
            break

        # Wait instead of time.sleep so the
        # thread can stop immediately.
        _polling_stop_event.wait(
            interval
        )

    print(
        "[complaint_handler] "
        "Email polling stopped."
    )


# ============================================================
# BACKGROUND POLLING
# ============================================================

def start_email_polling_background():
    """
    Start Gmail polling in a background daemon thread.

    FastAPI calls this when the backend starts.

    If the backend process is stopped,
    the daemon thread automatically stops.
    """

    global _polling_thread

    # Already running
    if (
        _polling_thread is not None
        and _polling_thread.is_alive()
    ):

        print(
            "[complaint_handler] "
            "Polling already running."
        )

        return

    _polling_stop_event.clear()

    _polling_thread = threading.Thread(
        target=start_email_polling,
        kwargs={
            "run_once": False
        },
        daemon=True,
        name="complaint-email-poller",
    )

    _polling_thread.start()

    print(
        "[complaint_handler] "
        "Background complaint polling started."
    )


# ============================================================
# STOP BACKGROUND POLLING
# ============================================================

def stop_email_polling():
    """
    Stop the background polling thread.
    """

    _polling_stop_event.set()

    print(
        "[complaint_handler] "
        "Stopping email polling..."
    )


# ============================================================
# MANUAL TEST / CLI
# ============================================================

if __name__ == "__main__":

    from src.complaints.complaint_event import (
        fetch_new_complaint_emails,
        extract_complaint_from_audio,
    )

    print("=" * 60)
    print("COMPLAINT HANDLER")
    print("=" * 60)

    print(
        "1. Process new complaint emails once"
    )

    print(
        "2. Process audio complaint"
    )

    print(
        "3. Approve pending reply"
    )

    print(
        "4. Reject pending reply"
    )

    print(
        "5. Start continuous polling"
    )

    print(
        "6. Exit"
    )

    while True:

        choice = input(
            "\nChoose an option [1-6]: "
        ).strip()

        # ----------------------------------------------------
        # Option 1
        # ----------------------------------------------------

        if choice == "1":

            poll_complaint_emails_once()

        # ----------------------------------------------------
        # Option 2
        # ----------------------------------------------------

        elif choice == "2":

            audio_path = input(
                "Path to audio file: "
            ).strip()

            complaint = (
                extract_complaint_from_audio(
                    audio_path
                )
            )

            handle_audio_complaint(
                complaint
            )

        # ----------------------------------------------------
        # Option 3
        # ----------------------------------------------------

        elif choice == "3":

            cid = input(
                "Customer ID to approve: "
            ).strip()

            approve_and_send_reply(
                cid,
                approve=True,
            )

        # ----------------------------------------------------
        # Option 4
        # ----------------------------------------------------

        elif choice == "4":

            cid = input(
                "Customer ID to reject: "
            ).strip()

            approve_and_send_reply(
                cid,
                approve=False,
            )

        # ----------------------------------------------------
        # Option 5
        # ----------------------------------------------------

        elif choice == "5":

            try:

                start_email_polling()

            except KeyboardInterrupt:

                stop_email_polling()

                print(
                    "\nPolling stopped."
                )

        # ----------------------------------------------------
        # Option 6
        # ----------------------------------------------------

        elif choice == "6":

            print(
                "Exiting."
            )

            break

        else:

            print(
                "Enter a number 1-6."
            )