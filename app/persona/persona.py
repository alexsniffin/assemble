class Persona:
    """ Represents a persona for the agent, providing specific personality traits and prompts. """

    def __init__(self, description: str):
        self.description = description

    def prompt(self) -> str:
        """ Generate a persona-specific prompt. """
        return self.description
