# tests/fixtures/documents.py

"""
Controlled Document Knowledge Base for API and E2E Tests.
Contains plain string constants to be uploaded or ingested by tests.
No application logic or pytest fixtures belong here.
"""

# ==========================================
# 1. Standard Factual Policy (Plain Text)
# ==========================================
# Used for testing basic retrieval, evaluation grounding, and conversational memory.
# Key Facts: 
# - Leave: 24 days
# - Remote: 3 days/week
# - Probation: 6 months
# - Learning Budget: 50,000 INR

NOVATECH_POLICY_TEXT = """
NovaTech Employee Policy

NovaTech employees receive 24 paid leave days per year.

Remote employees may work from home three days per week.

The standard probation period for new employees is six months.

Employees receive an annual learning budget of 50,000 INR.
"""


# ==========================================
# 2. Multi-Context Benefits (Plain Text)
# ==========================================
# Used for testing queries that require pulling multiple chunks together.
# Key Facts:
# - Health Insurance: All full-time employees
# - Wellness Allowance: 12,000 INR/year
# - Sick Leave: 10 days/year
# - Parental Leave: 16 weeks

NOVATECH_BENEFITS_TEXT = """
NovaTech Employee Benefits

NovaTech provides health insurance to all full-time employees.

Employees receive a wellness allowance of 12,000 INR per year.

The company provides 10 days of paid sick leave annually.

Parental leave is available for 16 weeks.
"""


# ==========================================
# 3. Security Policy (Markdown)
# ==========================================
# Used for testing the multi-format LoaderFactory (Markdown extraction).
# Key Facts:
# - Passwords: 12 characters min
# - MFA: Mandatory
# - Training: Every six months

NOVATECH_SECURITY_MARKDOWN = """
# NovaTech Security Policy

## Password Policy

Employees must use passwords with at least 12 characters.

## Multi-Factor Authentication

Multi-factor authentication is mandatory for all company accounts.

## Security Training

Employees must complete security training every six months.
"""