### OWS providers and services.

from .base import List, Url


class OwsOperation:
    def __init__(self):
        self.name = ''
        self.formats: List[str] = []
        self.get_url: Url = ''
        self.post_url: Url = ''
        self.parameters: dict = {}
