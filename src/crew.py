from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import tools


@CrewBase
class WebscraperCrew:
    """webscraper crew"""

    # Agent definitions
    @agent
    def scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["scraper"],
            tools=[
                tools.web_scrape,
            ],  
            verbose=True,
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],
            tools=[], 
            verbose=True,
        )

    # Task definitions
    @task
    def scape_site(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_site"],
        )

    @task
    def summarize_site(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_site"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Test crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
