#!/usr/bin/env python3
import difflib
import re
from typing import List

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
BASE_EMAIL_REGEX = r"[a-z0-9_.+-]+@[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z]{2,})+"

# EMAIL_REGEX for full validation (anchors added)
EMAIL_REGEX = f"^{BASE_EMAIL_REGEX}$"
email_pattern = re.compile(EMAIL_REGEX, re.IGNORECASE)

# EXTRACTION_REGEX (derived from the base pattern without anchors)
EXTRACTION_REGEX = BASE_EMAIL_REGEX
extraction_pattern = re.compile(EXTRACTION_REGEX, re.IGNORECASE)

# Manually sorted list of common domains to correct typical typos.
COMMON_DOMAINS = [
    "aol.com",
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "live.com",
    "outlook.com",
    "yahoo.com",
]

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
        # Dictionary of common top-level domain (TLD) typos mapped to their correct forms.
        # These corrections help fix frequent user errors when entering email addresses.
        tld_corrections = {
            ".cm": ".com",
            ".co.k": ".co.uk",
            ".co.ku": ".co.uk",
            ".co.ulk": ".co.uk",
            ".coo": ".co",
            ".coom": ".com",
            ".comm": ".com",
            ".con": ".com",
            ".cmo": ".com",
            ".cpm": ".com",
            ".edu.": ".edu",
            ".netl": ".net",
            ".ney": ".net",
            ".nrt": ".net",
            ".ocm": ".com",
            ".ogr": ".org",
            ".ogm": ".org",
            ".prg": ".org",
            ".vom": ".com",
            ".xom": ".com",
        }

    original_email = email  # Save original for error reporting
    email = email.strip().lower()

    # Check for missing '@'
    if "@" not in email:
        # Try to fix by checking if it ends with a common domain.
        matched = False
        for domain in COMMON_DOMAINS:
            if email.endswith(domain):
                local_part = email[: -len(domain)]
                local_part = local_part.rstrip(".-")
                email = f"{local_part}@{domain}"
                matched = True
                break
        if not matched:
            print(
                f"Error: {original_email} (missing '@' symbol and unrecognized domain)"
            )
            return original_email

    # Check for too many '@' symbols.
    if email.count("@") > 1:
        # Candidate 1: Replace consecutive '@@' with a single '@'
        candidate1 = email.replace("@@", "@")
        if candidate1.count("@") == 1 and email_pattern.fullmatch(candidate1):
            email = candidate1
        else:
            # Candidate 2: Split at the first '@' and join the remaining parts with '.'
            parts = email.split("@")
            candidate2 = parts[0] + "@" + ".".join(parts[1:])
            if candidate2.count("@") == 1 and email_pattern.fullmatch(candidate2):
                email = candidate2
            else:
                print(f"Error: {original_email} (too many '@' symbols, unable to fix)")
                return original_email

    local_part, domain_part = email.split("@", 1)

    # Normalize the local part: remove multiple consecutive dots and trim any leading/trailing dots.
    local_part = re.sub(r"\.{2,}", ".", local_part).strip(".")

    # Apply TLD corrections.
    for typo, correct in tld_corrections.items():
        if domain_part.endswith(typo):
            domain_part = domain_part[: -len(typo)] + correct
            break

    # If the domain starts or ends with a hyphen, print error and attempt to fix by removing it.
    if domain_part.startswith("-") or domain_part.endswith("-"):
        print(f"Error: {original_email} (invalid domain: starts or ends with a hyphen)")
        fixed_domain = domain_part.strip("-")
        # Double-check the fixed domain using a simple regex.
        domain_regex = re.compile(
            r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z]{2,})+$", re.IGNORECASE
        )
        if not fixed_domain or not domain_regex.fullmatch(fixed_domain):
            return original_email
        domain_part = fixed_domain

    # Check for two consecutive periods in the domain.
    if ".." in domain_part:
        print(f"Error: {original_email} (domain has consecutive periods)")
        candidate_domain = re.sub(r"\.{2,}", ".", domain_part)
        candidate_email = f"{local_part}@{candidate_domain}"
        # Validate the candidate email using the precompiled email_pattern.
        if candidate_email.count("@") == 1 and email_pattern.fullmatch(candidate_email):
            domain_part = candidate_domain
        else:
            return original_email

    # Suggest a close match for the domain if possible.
    close_matches = difflib.get_close_matches(
        domain_part, COMMON_DOMAINS, n=1, cutoff=0.9
    )
    if close_matches:
        domain_part = close_matches[0]

    fixed_email = f"{local_part}@{domain_part}"
    return fixed_email


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


def fix_changed(email: str, suggested_fix: str) -> bool:
    """
    Check if the suggested fix is different from the original email, ignoring case differences.
    """
    if not email or not suggested_fix:
        return False  # Handle edge cases where either is empty

    return email.strip().lower() != suggested_fix.strip().lower()


def get_unique_sorted_values(
    df: pd.DataFrame, columns: List[str], column_name: str = "Email"
) -> pd.DataFrame:
    """
    Extracts unique values from the specified columns in a DataFrame,
    sorts them alphabetically based on their string representation,
    and returns them as a DataFrame with two columns:
      - one for the value (named by column_name, default "Email")
      - one for the source column(s) from which the value was extracted.
    If a value is found in multiple columns, the source names are comma-separated.
    """
    # Check that all specified columns exist.
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {', '.join(missing_columns)}"
        )

    # Dictionary mapping each email value to a set of source columns where it was found.
    email_sources = {}
    for col in columns:
        # Extract non-null values as strings from the current column.
        values = df[col].dropna().astype(str)
        for value in values:
            # Update or initialize the set of sources for each email.
            email_sources.setdefault(value, set()).add(col)

    # Sort the unique email values.
    sorted_emails = sorted(email_sources.keys())

    # Build the output data: each email paired with its source(s) (comma-separated if multiple).
    data = [
        {column_name: email, "Source": ", ".join(sorted(email_sources[email]))}
        for email in sorted_emails
    ]

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Sample list of emails, including some with common errors and disposable domains.
    emails = [
        ".badlocal@domain.com",
        "bad..local@domain.com",
        "badlocal.@domain.com",
        "customer@hotmail.cmo",
        "GOOD.EMAIL123@SUB.DOMAIN.NET",
        "INVALID-EMAIL@@example.com",
        "missingatsymbol.com",
        "no-tld@domain",
        "temp@mailinator.com",
        "temporary@10minutemail.com",
        "test@guerrillamail.com",
        "test.email+spam@gmail.com",
        "unknown@dispostable.com",
        "user@-domain.com",
        "user@domain-.com",
        "user@domain",
        "user@gnail.com",
        "user@sub..domain.com",
        "user@yopmail.com",
        "Valid.Email@Example.com",
        "wrong_domain@.com",
    ]

    # Create a DataFrame with the sample emails.
    df = pd.DataFrame({"Email": emails})

    # Vectorized email validation using the precompiled regex pattern.
    df["Is_Valid"] = (
        df["Email"].str.strip().apply(lambda x: bool(email_pattern.fullmatch(x)))
    )

    # Suggest fixes for emails that might be slightly off.
    df["Suggested_Fix"] = df["Email"].apply(suggest_email_fix)

    # See if the Suggested_Fix is different than the Email other than casing.
    df["Fix_Changed"] = df.apply(
        lambda row: fix_changed(row["Email"], row["Suggested_Fix"]), axis=1
    )

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

    print(df.info())
    df.to_csv("Emails_Checked.csv", index=False)

    # Get and test the unique list of emails.
    unique_emails_df = get_unique_sorted_values(df, ["Suggested_Fix", "Normalized"])
    unique_emails_df["Is_Valid"] = (
        unique_emails_df["Email"]
        .str.strip()
        .apply(lambda x: bool(email_pattern.fullmatch(x)))
    )

    print(unique_emails_df.info())
    unique_emails_df.to_csv("Unique_Emails.csv", index=False)
