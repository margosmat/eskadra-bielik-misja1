import os
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm

BIELIK_MODEL_NAME = os.getenv("BIELIK_MODEL_NAME", "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0")

GEMINI_MODEL_NAME="gemini-flash-latest"

pro_agent = Agent(
    name="pro_agent",
    model=LiteLlm(model=f"ollama_chat/{BIELIK_MODEL_NAME}"),
    description=("""Agent odpowiedzialny za obronienie tezy historycznej zawartej w prompcie."""),
    instruction=("""
                Jesteś super znawcą historii Polski. Na podstawie zadanej tezy dostarcz argumentów, które by ją popierały.
                """)
)

con_agent = Agent(
    name="con_agent",
    model=LiteLlm(model=f"ollama_chat/{BIELIK_MODEL_NAME}"),
    description=("""Agent odpowiedzialny za obalenie tezy historycznej zawartej w prompcie."""),
    instruction=("""
                Jesteś super znawcą historii Polski. Na podstawie zadanej tezy dostarcz argumentów, które by ją obalały.
                """)
)

parallel_agent = ParallelAgent(
    name="content_creator_agent",
    description=(
        """
        Agent uruchamiający równolegle dwóch pod-agentów, którzy mają rozwinąć temat pod kątem historycznym.
        """),
    sub_agents=[pro_agent, con_agent]
)

verifier_agent = Agent(
    name="verifier",
    model=GEMINI_MODEL_NAME,
    description=("""Jesteś agentem, który ma porównać treści obu pod-agentów i stwierdzić, czy teza opsiywana przez con-agenta miałąby szansę się sprawdzić."""),
    instruction=(
        """
            Podsumuj tezy pod-agentów i oceń czy alternatywna historia opisywana przez con-agenta mogłaby się sprawdzić.
        """
    )
)

root_agent = SequentialAgent(
    name="root",
    description=("""Agent, który nadzoruje agentów historycznych"""),
    sub_agents=[parallel_agent, verifier_agent]
)