#!/usr/bin/env python3
import difflib
import re

import pandas as pd

# Optional: For email validation, install email-validator: pip install email-validator
try:
    from email_validator import EmailNotValidError, validate_email
except ImportError:
    validate_email = None
    EmailNotValidError = None


# Optional: For MX record checking, install dnspython: pip install dnspython
try:
    import dns.resolver
except ImportError:
    dns = None

# Original regex (allows uppercase, kept for reference)
# EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'

# Updated base pattern used for both validation and extraction.
# This new regex ensures that the domain does not start or end with a hyphen.
BASE_EMAIL_REGEX = r"[a-z0-9_.+-]+@[a-z0-9]([a-z0-9-]*[a-z0-9])?\.[a-z]{2,}"

# EMAIL_REGEX for full validation (anchors added)
EMAIL_REGEX = f"^{BASE_EMAIL_REGEX}$"
email_pattern = re.compile(EMAIL_REGEX, re.IGNORECASE)

# EXTRACTION_REGEX (derived from the base pattern without anchors)
EXTRACTION_REGEX = BASE_EMAIL_REGEX
extraction_pattern = re.compile(EXTRACTION_REGEX, re.IGNORECASE)

# Manually sorted list of common domains to correct typical typos.
COMMON_DOMAINS = ["gmail.com", "hotmail.com", "icloud.com", "outlook.com", "yahoo.com"]

# Manually sorted list of known disposable email domains (expanded with yopmail.com)
DISPOSABLE_DOMAINS = [
    "10minutemail.com",
    "guerrillamail.com",
    "mailinator.com",
    "trashmail.com",
    "yopmail.com",
]


def is_valid_email(email: str) -> bool:
    """
    Check if an email is valid using a precompiled regex (full match).
    """
    email = email.strip()
    return bool(email_pattern.fullmatch(email))


def is_valid_email_lib(email: str) -> bool:
    """
    Alternative email validation using the email_validator library.

    If email-validator is not installed, returns False.
    """
    if validate_email is None:
        print("Validation unavailable (email-validator not installed)")
        return False

    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False


def suggest_email_fix(email: str, tld_corrections: dict = None) -> str:
    if tld_corrections is None:
        tld_corrections = {".cmo": ".com", ".con": ".com"}

    email = email.strip().lower()

    # Handle missing '@' symbol by checking if the email ends with one of the common domains.
    if email.count("@") == 0:
        for domain in COMMON_DOMAINS:
            if email.endswith(domain):
                local_part = email[: -len(domain)]
                # Optionally, strip trailing punctuation or unwanted characters from local_part.
                local_part = local_part.rstrip(".-")
                email = f"{local_part}@{domain}"
                break
        else:
            return "Invalid format (missing '@' symbol and unrecognized domain)"
    elif email.count("@") > 1:
        return "Invalid format (too many '@')"

    local_part, domain_part = email.split("@")

    # Normalize the local part: remove multiple consecutive dots and strip leading/trailing dots.
    if ".." in local_part:
        local_part = re.sub(r"\.{2,}", ".", local_part)
    local_part = local_part.strip(".")

    # Correct common TLD typos using the provided dictionary.
    domain_part = next(
        (
            domain_part[: -len(typo)] + correct
            for typo, correct in tld_corrections.items()
            if domain_part.endswith(typo)
        ),
        domain_part,
    )

    # Validate domain structure: ensure the domain does not start or end with a hyphen.
    if domain_part.startswith("-") or domain_part.endswith("-"):
        return "Invalid domain (starts/ends with a hyphen)"

    # Use difflib to suggest a close match for common domains.
    close_matches = difflib.get_close_matches(
        domain_part, COMMON_DOMAINS, n=1, cutoff=0.7
    )
    if close_matches:
        domain_part = close_matches[0]

    return f"{local_part}@{domain_part}"


def extract_emails(text: str) -> list:
    """
    Extract all email addresses from a block of text.
    """
    return extraction_pattern.findall(text)


def normalize_email(email: str) -> str:
    """
    Normalize an email address by stripping whitespace, lowercasing,
    and normalizing the local part (e.g., removing extra dots).
    """
    email = email.strip().lower()
    if email.count("@") != 1:
        return email  # Return as is if the format is not proper

    local_part, domain_part = email.split("@")
    local_part = re.sub(r"\.{2,}", ".", local_part).strip(".")
    return f"{local_part}@{domain_part}"


def get_email_components(email: str) -> dict:
    """
    Split an email into its local and domain parts.
    Returns a dictionary with keys 'local' and 'domain'.
    """
    email = normalize_email(email)
    if email.count("@") == 1:
        local_part, domain_part = email.split("@")
        return {"local": local_part, "domain": domain_part}
    return {"error": "Invalid email format"}


def is_disposable_email(email: str) -> bool:
    """
    Check if the email domain is from a known disposable email provider.
    """
    components = get_email_components(email)
    if "error" in components:
        return False
    domain = components["domain"]
    return domain in DISPOSABLE_DOMAINS


def has_valid_mx(email: str):
    """
    Check if the email domain has valid MX records.
    Requires dnspython. If dnspython is not installed, returns a message.
    """
    if dns is None:
        return "MX check unavailable (dnspython not installed)"

    components = get_email_components(email)
    if "error" in components:
        return False
    domain = components["domain"]
    try:
        answers = dns.resolver.resolve(domain, "MX")
        return bool(answers)
    except Exception:
        return False


if __name__ == "__main__":
    # Sample list of emails, including some with common errors and disposable domains.
    emails = [
        "Valid.Email@Example.com",
        "INVALID-EMAIL@@example.com",
        "missingatsymbol.com",
        "wrong_domain@.com",
        "GOOD.EMAIL123@SUB.DOMAIN.NET",
        "user@gnail.com",
        "customer@hotmail.cmo",
        "bad..local@domain.com",
        ".badlocal@domain.com",
        "badlocal.@domain.com",
        "user@-domain.com",
        "user@domain-.com",
        "temp@mailinator.com",
        "user@yopmail.com",
    ]

    # Create a DataFrame with the sample emails.
    df = pd.DataFrame({"Email": emails})

    # Vectorized email validation using the precompiled regex pattern.
    df["Is_Valid"] = (
        df["Email"].str.strip().apply(lambda x: bool(email_pattern.fullmatch(x)))
    )

    # Suggest fixes for emails that might be slightly off.
    df["Suggested_Fix"] = df["Email"].apply(suggest_email_fix)

    # Additional validation using the email_validator library for comparison.
    df["Is_Valid_Lib"] = df["Email"].apply(is_valid_email_lib)

    # Normalize emails.
    df["Normalized"] = df["Email"].apply(normalize_email)

    # Extract local and domain components.
    df["Components"] = df["Email"].apply(get_email_components)

    # Check for disposable email domains.
    df["Is_Disposable"] = df["Email"].apply(is_disposable_email)

    # Check if the email's domain has valid MX records (if dnspython is installed).
    df["Has_Valid_MX"] = df["Email"].apply(has_valid_mx)

    print(df)
