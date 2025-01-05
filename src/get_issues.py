from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from pydantic import Field
from typing import Dict, Any
import requests
import os
import argparse
import json

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def fetch_github_issues(repo, max_issues=5):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "open", "per_page": max_issues}
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch issues from {repo}: {response.status_code} {response.text}"
        )

    issues = response.json()
    return issues


def fetch_issue_comments(issue):
    comments_url = issue.get("comments_url", "")
    comments_response = requests.get(comments_url, headers=HEADERS)
    if comments_response.status_code != 200:
        print("warning: comments_response.status_code", comments_response.status_code)
        return ""
    else:
        comments = comments_response.json()

    comments_text = "\n".join(
        [comment.get("body", "No comment") for comment in comments]
    )
    return comments_text


def strip_body(raw_body):
    to_strip = [
        "-->",
        "<!--",
        "<!-- A clear and concise description of the Problem. -->",
        "A clear and concise description of what you expected to happen.",
        "Steps to reproduce the behavior:",
        "Turn off brightness on Room 300 @ Cross Timbers",
        "XX is requesting that we turn down the brightness on the control box because he",
        "1. Go to '...'",
        "2. Click on '....'",
        "3. Scroll down to '....'",
        "4. See error",
        "If applicable, add screenshots to help explain your problem.",
        "- iOS/Android: [e.g. vX.X.X]",
        "- Device Type: [e.g. iPhone6]",
        '- Cloud: [e.g. vX.X.X or "master"]',
        "- Edge*:",
        "- Nurse UI*:",
        "Add any other context about the problem here.",
        "Deployments affected: latest-v3, v3-staging....",
        "- Seen in accounts: dev, prod...",
        "## Prb-Risk Decision Tree",
        "Major",
        "- [ ] total absence in the fulfillment of a requirement and no existing control measures in place for a system primary function (e.g. requirements related to bed exit or reposition notification), AND"
        "- [ ] directly or indirectly places a patient at risk of injury or harm",
        '(if both of the above are checked, this is a "major" issue)',
        "Moderate",
        "- [ ] non-safety related or interpretation-related problem",
        '(if checked this is a "moderate" risk issue)',
        "Minor",
        "Enhancement",
        "- [ ] problem for which correction is not required, but shall be retained for future monitoring and consideration",
        '(if checked this is a "minor" issue)',
        "Other",
        "- [ ] Deemed not a problem due to where the issue was found or not matching one of the above categories. If categorized this way, the issue should be removed from the problem board.",
    ]

    stripped = ""
    for line in raw_body.split("\n"):
        do_strip = False
        for strip in to_strip:
            if strip in line:
                do_strip = True
                break

        if not do_strip:
            stripped += line

    return stripped



def write_issues(issues):
    issues_data = {}
    for issue in issues:
        issue_number = issue.get("number", "No number")
        issue_title = issue.get("title", "No title")
        issue_body = strip_body(issue.get("body", "No description"))
        issue_labels = [label["name"] for label in issue.get("labels", [])]
        comments_text = fetch_issue_comments(issue)

        issue_data = {
            "issue_number": issue_number,
            "title": issue_title,
            "body": issue_body,
            "labels": issue_labels,
            "comments": comments_text,
        }

        issues_data[issue_number] = issue_data

    with open(f"./knowledge/issue_data.json", "w") as f:
        json.dump(issues_data, f)


def retrieve_issues():
    repo_name = "cognitohealth/universe"
    issues = fetch_github_issues(repo_name)
    write_issues(issues)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Issues Script")
    parser.add_argument(
        "--retrieve-issues",
        action="store_true",
        help="Retrieve issues from GitHub and write to a file",
    )

    args = parser.parse_args()

    if args.retrieve_issues:
        retrieve_issues()
    else:
        parser.print_help()
