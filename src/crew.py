from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from composio_crewai import ComposioToolSet, Action

toolset = ComposioToolSet(api_key="8rl2ei1l0yqskegifhps5")


@CrewBase
class GithubCrew:
    # Agent definitions
    @agent
    def github(self) -> Agent:
        return Agent(
            config=self.agents_config["github"],
            tools=toolset.get_tools(actions=[Action.GITHUB_LIST_PULL_REQUESTS]),
            verbose=True,
        )

    # @agent
    # def summarizer(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["summarizer"],
    #         tools=[],
    #         verbose=True,
    #     )

    # Task definitions
    @task
    def get_pull_requests(self) -> Task:
        return Task(
            config=self.tasks_config["get_pull_requests"],
        )

    # @task
    # def summarize_site(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["summarize_pull_requests"],
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the Test crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            max_retry_limit=0,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
