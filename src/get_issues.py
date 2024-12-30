from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from pydantic import Field
from typing import Dict, Any
import requests
import os
import argparse
import yaml

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
    """
    Write each issue to a file in ./data/issues/<issue_number>.txt
    The file should use yml where the top-level keys are issue_number, title, body, labels, comments.

    """
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

        # make sure the directory exists
        os.makedirs("./data/issues", exist_ok=True)

        with open(f"./data/issues/{issue_number}.yml", "w") as f:
            yaml.dump(issue_data, f)


def retrieve_issues():
    repo_name = "cognitohealth/universe"
    issues = fetch_github_issues(repo_name)
    write_issues(issues)
    # knowledge_content = build_knowledge_content(issues)

    # print(knowledge_content.split("\n")[0:1000])
    # with open("issues.txt", "w") as f:
    #     f.write(knowledge_content)

class GitHubIssuesKnowledgeSource(BaseKnowledgeSource):
    """Knowledge source that fetches GitHub issues."""

    def load_content(self) -> Dict[Any, str]:
        """Load issues from file system"""
        issues = []
        for file in os.listdir("./data/issues"):
            if file.endswith(".yml"):
                with open(f"./data/issues/{file}", "r") as f:
                    issue = yaml.safe_load(f)
                    issues.append(issue)

        formatted_data = self._format_issues(issues)
        return {"GitHub Issues": formatted_data}

    def _format_issues(self, issues: list) -> str:
        """Format issues into readable text."""
        formatted = "GitHub Issues:\n\n"
        for issue in issues:
            formatted += f"""
                Title: {issue['title']}
                Number: {issue['issue_number']}
                Body: {issue['body']}
                Labels: {issue['labels']}
                Comments: {issue['comments']}
                -------------------"""
        return formatted

    def add(self) -> None:
        """Process and store the issues."""
        content = self.load_content()
        for _, text in content.items():
            chunks = self._chunk_text(text)
            self.chunks.extend(chunks)

        self._save_documents()

def run_crew():
    issues_source = GitHubIssuesKnowledgeSource()
    llm = LLM(model="gpt-4o", temperature=0)

    agent = Agent(
        role="Project manager",
        goal="Provide insights about GitHub issues.",
        backstory="You are an expert in understanding and summarizing GitHub issues that represent the current state of the product. You excel at providing a good summary of what needs to be addressed in the company to CEOs and executives",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        knowledge_sources=[issues_source],
    )

    task = Task(
        description="Answer the following questions about GitHub issues: {question}",
        expected_output="A brief answer based on the provided issues",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
    )

    result = crew.kickoff(
        inputs={"question": "Summarize the issues related to EC Services"}
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Issues Script")
    parser.add_argument(
        "--retrieve-issues",
        action="store_true",
        help="Retrieve issues from GitHub and write to a file",
    )
    parser.add_argument("--run", action="store_true", help="Run the CrewAI crew method")

    args = parser.parse_args()

    if args.retrieve_issues:
        retrieve_issues()
    elif args.run:
        run_crew()
    else:
        parser.print_help()
