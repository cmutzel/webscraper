from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
import os
import yaml
from typing import Dict, Any


json_source = JSONKnowledgeSource(file_paths=["issue_data.json"])

# knowledge = Knowledge(
#     collection_name="json_knowledge",
#     sources=[json_source]
# )

# print(knowledge.query("Get all issues related to 'bug'"))

@CrewBase
class GithubCrew:
    # Agent definitions
    @agent
    def issue_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["issue_summarizer"],
            tools=[],
            verbose=True,
        )

    # Task definitions
    @task
    def summarize_issues(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_issues"],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Test crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            max_retry_limit=0,
            knowledge_sources=[json_source],
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
