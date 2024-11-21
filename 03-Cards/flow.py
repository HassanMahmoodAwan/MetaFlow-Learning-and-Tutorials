from metaflow import FlowSpec, Parameter, step
import requests, pandas, string

URL = "https://upload.wikimedia.org/wikipedia/commons/4/45/Blue_Marble_rotating.gif"

class FancyDefaultCardFlow(FlowSpec):

    image_url = Parameter('image_url', default=URL)

    @step
    def start(self):
        self.image = requests.get(self.image_url,
                                  headers={'user-agent': 'metaflow-example'}).content
        self.dataframe = pandas.DataFrame({'lowercase': list(string.ascii_lowercase),
                                           'uppercase': list(string.ascii_uppercase)})
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    FancyDefaultCardFlow()