# src/complaints/complaint_handler.py
"""
Complaint Handler -- "Looks up customer, triggers scoring now" in the
system diagram.

Takes a ComplaintEvent (from complaint_event.py) and:

  EMAIL complaints
  ----------------
  1. If CustomerID/contact number is missing entirely -> auto-reply
     asking the customer to provide it.
  2. If a CustomerID WAS given but doesn't match any account -> a
     different auto-reply asking them to double-check it (distinct from
     case 1, since they already tried to help us).
  3. Otherwise -> look up the customer, run the FULL predictor pipeline
     (risk score + SHAP reasons + GenAI retention offer, via
     src/prediction/predictor.py -- reused, not reimplemented), and save
     a DRAFT reply as "pending_approval". The offer email is NOT sent yet.
  4. A human (retention ops) reviews the draft and calls
     approve_and_send_reply(customer_id, approve=True) -- ONLY THEN does
     the actual offer email go out. This is an explicit organisational
     approval gate, separate from and in addition to the AI Guardrail
     agent's own approval inside predictor.py -- an AI-approved offer
     still can't reach a customer's inbox without a human sign-off.

  AUDIO complaints
  ----------------
  No customer-facing channel to reply through (it's an uploaded call
  recording, not a live conversation), so this handler just looks the
  customer up, triggers scoring, and logs the risk + reasons + retention
  strategy for the ops team to act on manually. If the customer can't be
  identified from the transcript, it's queued for manual review instead
  of guessing.

  AUTO-POLLING
  ------------
  Option 6 in the menu below starts a loop that calls
  fetch_new_complaint_emails() every `complaints.poll_interval_seconds`
  (config.yaml) automatically -- this is what makes the email channel
  actually real-time instead of only running when someone manually picks
  option 1.

CAVEAT (please read): the Telco_customer_churn.xlsx dataset used across
this project has NO actual phone/contact-number column (only a
"Phone Service" Yes/No flag) -- see src/data/loader.py's required_columns.
So contact-number lookup below is a placeholder that will not find a real
match against this dataset. In production this should query a separate
CRM/customer-contact table; the ComplaintEvent/handler interface is
already shaped to support that once such a table exists.
"""

import os
import json
import time
import smtplib
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

# Load .env HERE too -- SMTP_HOST/USER/PASSWORD are read directly in this
# file's _send_email(), and this module can be run/imported on its own
# without necessarily going through complaint_event.py's audio path first.
load_dotenv()


# --------------------------------------------------
# Pending-approval store
# --------------------------------------------------
# A simple JSON file standing in for a real "approvals" table/dashboard.
# Keyed by CustomerID. Holds the drafted reply until a human approves it.

PENDING_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "pending_complaint_replies.json"
MANUAL_REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "manual_review_queue.json"


def _load_json_store(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_json_store(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# --------------------------------------------------
# Customer lookup
# --------------------------------------------------

def find_customer_row(customer_id: Optional[str], contact_number: Optional[str]) -> Optional[dict]:
    """
    Looks up a customer by CustomerID (works against the real dataset) or
    contact number (placeholder -- see module caveat above). Returns a
    dict of raw column -> value (same schema predictor.predict_customer()
    expects), or None if not found.
    """
    df = load_raw_data()
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()

    if customer_id:
        match = df[df["CustomerID"] == str(customer_id).strip().upper()]
        if not match.empty:
            return match.iloc[0].to_dict()

    if contact_number:
        # TODO: this dataset has no phone-number column to match against.
        # Wire this up to a real CRM/customer-contact table when available.
        # Left in place so the calling code doesn't need to change later.
        pass

    return None


# --------------------------------------------------
# Email sending (SMTP) -- shared by both reply types
# --------------------------------------------------

def _send_email(to_email: str, subject: str, body: str) -> None:
    """
    Sends a plain-text email via SMTP. Requires in .env (see .env.example):
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM_EMAIL
    SMTP_PASSWORD should be an app password (e.g. Gmail App Password).
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL", user)

    if not all([host, user, password]):
        raise EnvironmentError(
            "SMTP_HOST, SMTP_USER, and SMTP_PASSWORD must be set in .env "
            "to send reply emails. See .env.example."
        )

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(from_email, [to_email], msg.as_string())


def _reply_subject(original_subject: Optional[str]) -> str:
    original_subject = original_subject or "your message to us"
    return original_subject if original_subject.lower().startswith("re:") else f"Re: {original_subject}"


# --------------------------------------------------
# EMAIL complaint flow
# --------------------------------------------------

def send_missing_info_request(complaint: ComplaintEvent) -> dict:
    """
    Case 1: no CustomerID or contact number could be extracted at all.
    Sent immediately -- no organisational approval needed, since this
    isn't an offer, just a request for information.
    """
    body = (
        "Hi,\n\n"
        "Thanks for reaching out. To look into this for you, could you please "
        "reply with your Customer ID (found on your bill/account page) or the "
        "phone number registered on your account?\n\n"
        "Once we have that, our team will review your account and get back to you.\n\n"
        "Thanks,\nCustomer Care Team"
    )
    _send_email(complaint.sender_email, _reply_subject(complaint.subject), body)
    return {"status": "info_requested", "sender_email": complaint.sender_email}


def send_id_not_found_request(complaint: ComplaintEvent) -> dict:
    """
    Case 2: a CustomerID (or contact number) WAS extracted, but it didn't
    match any account. Distinct message from send_missing_info_request --
    this customer already gave us something, so we acknowledge that and
    ask them to double-check it, instead of asking as if they gave us
    nothing.
    """
    body = (
        "Hi,\n\n"
        "Thanks for reaching out. We couldn't find an account matching the "
        f"details provided (Customer ID: {complaint.customer_id or 'not recognized'}). "
        "Could you please double-check your Customer ID and reply with the "
        "correct one? You can find it on your bill or account page.\n\n"
        "Thanks,\nCustomer Care Team"
    )
    _send_email(complaint.sender_email, _reply_subject(complaint.subject), body)
    return {"status": "id_not_found", "sender_email": complaint.sender_email}


def _draft_offer_reply_body(customer_id: str, predictor_result: dict) -> str:
    if predictor_result["agent_approved"] and predictor_result["offer"]:
        offer_text = predictor_result["offer"]
        return (
            "Hi,\n\n"
            "Thank you for getting in touch, and sorry to hear about your experience. "
            "We'd like to make things right. Based on your account, here's what we can offer:\n\n"
            f"{offer_text}\n\n"
            "If you'd like to go ahead with this, just reply to this email and we'll get it set up.\n\n"
            "Thanks,\nCustomer Care Team"
        )
    # Agent escalated or found no policy-supported offer -- don't invent one.
    return (
        "Hi,\n\n"
        "Thank you for getting in touch, and sorry to hear about your experience. "
        "Your case has been passed to a member of our retention team, who will "
        "review your account and reach out to you directly.\n\n"
        "Thanks,\nCustomer Care Team"
    )


def handle_email_complaint(complaint: ComplaintEvent) -> dict:
    """
    Main entry point for ONE email ComplaintEvent. Never sends an offer
    email itself -- always stops at "pending_approval" and waits for
    approve_and_send_reply() to actually send it.
    """
    if complaint.missing_customer_info:
        return send_missing_info_request(complaint)

    customer_row = find_customer_row(complaint.customer_id, complaint.contact_number)
    if customer_row is None:
        # An ID WAS given, it just didn't match -- different message
        # than "you gave us nothing".
        return send_id_not_found_request(complaint)

    # ---- Trigger scoring now (risk + SHAP reasons + GenAI offer) ----
    result = predict_customer(customer_row, run_agent=True)

    draft_body = _draft_offer_reply_body(complaint.customer_id, result)

    pending = _load_json_store(PENDING_STORE_PATH)
    pending[complaint.customer_id] = {
        "customer_id": complaint.customer_id,
        "sender_email": complaint.sender_email,
        "subject": _reply_subject(complaint.subject),
        "draft_body": draft_body,
        "churn_probability": result["churn_probability"],
        "risk_tier": result["risk_tier"],
        "agent_approved": result["agent_approved"],
        "agent_escalated": result["agent_escalated"],
        "status": "pending_approval",
        "created_at": datetime.now().isoformat(),
    }
    _save_json_store(PENDING_STORE_PATH, pending)

    print(
        f"\n[complaint_handler] Draft reply ready for {complaint.customer_id} "
        f"(risk tier: {result['risk_tier']}, churn probability: {result['churn_probability']:.2%}). "
        f"Awaiting organisational approval -- call "
        f"approve_and_send_reply('{complaint.customer_id}', approve=True) to send."
    )

    return {
        "status": "pending_approval",
        "customer_id": complaint.customer_id,
        "risk_tier": result["risk_tier"],
        "churn_probability": result["churn_probability"],
    }


def approve_and_send_reply(customer_id: str, approve: bool = True, approved_by: str = "ops") -> dict:
    pending = _load_json_store(PENDING_STORE_PATH)
    record = pending.get(customer_id)

    if record is None:
        print(f"[complaint_handler] No pending reply found for customer '{customer_id}'. "
              f"(They may not have complained yet, or the ID doesn't match any pending entry.)")
        return {"status": "not_found", "customer_id": customer_id}

    if not approve:
        record["status"] = "rejected"
        record["decided_by"] = approved_by
        record["decided_at"] = datetime.now().isoformat()
        pending[customer_id] = record
        _save_json_store(PENDING_STORE_PATH, pending)
        print(f"[complaint_handler] Reply for customer '{customer_id}' marked as REJECTED. No email sent.")
        return record

    _send_email(record["sender_email"], record["subject"], record["draft_body"])

    record["status"] = "approved_sent"
    record["decided_by"] = approved_by
    record["decided_at"] = datetime.now().isoformat()
    pending[customer_id] = record
    _save_json_store(PENDING_STORE_PATH, pending)

    print(f"[complaint_handler] Reply APPROVED and SENT to {record['sender_email']} for customer '{customer_id}'.")
    return record


# --------------------------------------------------
# AUDIO complaint flow
# --------------------------------------------------

def handle_audio_complaint(complaint: ComplaintEvent) -> Optional[dict]:
    """
    Main entry point for ONE audio ComplaintEvent. No customer-facing
    reply channel here -- this just triggers scoring and logs the result
    for the ops team, or queues for manual review if the customer can't
    be identified from the transcript.
    """
    if not complaint.is_complaint:
        print("[complaint_handler] No complaint keywords detected in audio -- skipping.")
        return None

    if complaint.missing_customer_info:
        review_queue = _load_json_store(MANUAL_REVIEW_PATH)
        entry_id = complaint.subject or datetime.now().isoformat()
        review_queue[entry_id] = {
            "source": "audio",
            "transcript": complaint.raw_text,
            "matched_keywords": complaint.matched_keywords,
            "reason": "Customer could not be identified from audio (no CustomerID or contact number found).",
            "queued_at": datetime.now().isoformat(),
        }
        _save_json_store(MANUAL_REVIEW_PATH, review_queue)
        print(f"[complaint_handler] Could not identify customer from audio -- queued for manual review ({entry_id}).")
        return {"status": "manual_review_required"}

    customer_row = find_customer_row(complaint.customer_id, complaint.contact_number)
    if customer_row is None:
        review_queue = _load_json_store(MANUAL_REVIEW_PATH)
        entry_id = complaint.customer_id or complaint.subject
        review_queue[entry_id] = {
            "source": "audio",
            "transcript": complaint.raw_text,
            "customer_id_heard": complaint.customer_id,
            "reason": f"CustomerID '{complaint.customer_id}' not found in records.",
            "queued_at": datetime.now().isoformat(),
        }
        _save_json_store(MANUAL_REVIEW_PATH, review_queue)
        print(f"[complaint_handler] CustomerID '{complaint.customer_id}' not found -- queued for manual review.")
        return {"status": "manual_review_required"}

    # ---- Trigger scoring now (risk + SHAP reasons + GenAI offer) ----
    result = predict_customer(customer_row, run_agent=True)

    print("\n" + "=" * 60)
    print(f"AUDIO COMPLAINT PROCESSED -- {complaint.customer_id}")
    print("=" * 60)
    print(f"Keywords detected : {complaint.matched_keywords}")
    print(f"Churn probability : {result['churn_probability']:.2%}")
    print(f"Risk tier         : {result['risk_tier']}")
    print(f"Revenue at risk   : ${result['annual_revenue_at_risk']:,.2f}")
    print("\nTop reasons for churn risk:")
    for d in result["top_risk_drivers"][:3]:
        print(f"  - {d['feature']} (impact: +{d['shap_value']})")
    print("\nSuggested retention strategy:")
    for i, action in enumerate(result["retention_strategy"], 1):
        print(f"  {i}. {action}")
    if result["agent_approved"]:
        print(f"\nGenAI offer (route to ops for outreach):\n{result['offer']}")
    elif result["agent_escalated"]:
        print(f"\nEscalated: {result.get('escalation_reason')}")
    print("=" * 60 + "\n")

    return {
        "status": "processed",
        "customer_id": complaint.customer_id,
        "risk_tier": result["risk_tier"],
        "churn_probability": result["churn_probability"],
    }


# --------------------------------------------------
# Auto-polling loop -- makes the email channel actually real-time
# --------------------------------------------------

def start_email_polling(run_once: bool = False) -> None:
    """
    Polls fetch_new_complaint_emails() every
    config.yaml -> complaints.poll_interval_seconds, and processes each
    new email automatically via handle_email_complaint().

    This is what turns "check the inbox when someone remembers to" into
    an always-on real-time trigger. Stop with Ctrl+C.

    run_once=True processes a single poll cycle then returns -- useful
    for a quick demo/test without leaving a loop running.
    """
    from src.complaints.complaint_event import fetch_new_complaint_emails

    interval = get("complaints", "poll_interval_seconds", default=60)

    print("=" * 60)
    print("COMPLAINT EMAIL AUTO-POLLING STARTED")
    print(f"Checking every {interval} seconds. Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        while True:
            new_emails = fetch_new_complaint_emails()
            if new_emails:
                print(f"\n[complaint_handler] Processing {len(new_emails)} new email(s)...")
                for complaint in new_emails:
                    handle_email_complaint(complaint)
            else:
                print("[complaint_handler] No new complaint emails this cycle.")

            if run_once:
                return

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[complaint_handler] Polling stopped.")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    from src.complaints.complaint_event import (
        fetch_new_complaint_emails,
        extract_complaint_from_audio,
    )

    print("=" * 60)
    print("COMPLAINT HANDLER")
    print("=" * 60)
    print("1. Process new complaint emails once (IMAP inbox)")
    print("2. Process an uploaded audio complaint file")
    print("3. Approve & send a pending reply")
    print("4. Reject a pending reply")
    print("5. Start auto-polling (real-time, runs until Ctrl+C)")
    print("6. Exit")

    while True:
        choice = input("\nChoose an option [1-6]: ").strip()

        if choice == "1":
            new_emails = fetch_new_complaint_emails()
            print(f"\nFound {len(new_emails)} new complaint email(s).")
            for complaint in new_emails:
                handle_email_complaint(complaint)

        elif choice == "2":
            audio_path = input("Path to audio file: ").strip()
            complaint = extract_complaint_from_audio(audio_path)
            handle_audio_complaint(complaint)

        elif choice == "3":
            cid = input("Customer ID to approve: ").strip()
            approve_and_send_reply(cid, approve=True)

        elif choice == "4":
            cid = input("Customer ID to reject: ").strip()
            approve_and_send_reply(cid, approve=False)

        elif choice == "5":
            start_email_polling()

        elif choice == "6":
            print("Exiting.")
            break

        else:
            print("Enter a number 1-6.")