class NamedQuery:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"SELECT parent_id FROM object as o WHERE name = '{self.name}'"

    def __eq__(self, other):
        return str(self) == str(other)

