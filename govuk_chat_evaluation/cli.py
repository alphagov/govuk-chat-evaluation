import click
from dotenv import load_dotenv

from .file_system import project_root
from . import jailbreak_guardrails
from . import output_guardrails
from . import question_router
from . import rag_answers
from . import retrieval
from . import topic_tagger

load_dotenv(project_root() / ".env.aws")
load_dotenv()


@click.group()
def main():
    """Command line interface to run evaluations of GOV.UK chat"""


main.add_command(jailbreak_guardrails.main)
main.add_command(output_guardrails.main)
main.add_command(question_router.main)
main.add_command(rag_answers.main)
main.add_command(retrieval.main)
main.add_command(topic_tagger.main)
