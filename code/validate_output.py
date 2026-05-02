import csv
import re
import sys

def main():
    import os
    if os.path.exists("../support_tickets/output.csv"):
        csv_path = "../support_tickets/output.csv"
    elif os.path.exists("support_tickets/output.csv"):
        csv_path = "support_tickets/output.csv"
    elif os.path.exists("hackerrank-orchestrate-may26/support_tickets/output.csv"):
        csv_path = "hackerrank-orchestrate-may26/support_tickets/output.csv"
    else:
        csv_path = "output.csv"
        
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    errors = []

    # Regex for unsupported emails/urls
    # Allow help@hackerrank.com, exclude support@hackerrank.com
    support_email_re = re.compile(r"support@hackerrank\.com", re.IGNORECASE)
    
    # "Couldn't find info" patterns
    not_found_re = re.compile(r"(could not find|couldn't find|no information|don't have information|unable to find)", re.IGNORECASE)

    for i, row in enumerate(rows):
        row_num = i + 2
        response = row.get("Response", "")
        
        # Check unsupported email
        if support_email_re.search(response):
            errors.append(f"Row {row_num}: Contains unsupported email support@hackerrank.com")

        # Check for 'couldn't find info' on keyword-matched docs
        if not_found_re.search(response):
            issue = row.get("Issue", "").lower()
            subject = row.get("Subject", "").lower()
            combined = f"{issue} {subject}"
            
            # Did they hit a keyword?
            keywords = ["model improvement", "privacy settings", "remove employee", "remove user", "inactivity", "bug bounty"]
            if any(kw in combined for kw in keywords):
                errors.append(f"Row {row_num}: Says couldn't find info but matches keyword boosting docs.")

    if errors:
        print("Validation Failed:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("Validation Passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
