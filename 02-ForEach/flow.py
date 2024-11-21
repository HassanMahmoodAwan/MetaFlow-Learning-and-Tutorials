from metaflow import FlowSpec, step

class ForeachFlow(FlowSpec):

    @step
    def start(self):
        self.names = "Hassan"
        self.titles = ['Stranger Things',
                       'House of Cards',
                       'Narcos']
        self.next(self.a, foreach='titles')

    @step
    def a(self):
        self.title = self.input + "input is titles"
        print(self.names)
        self.names = self.names
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [input.title for input in inputs]
        for inp in inputs:
            print(inp.names)
        self.next(self.end)

    @step
    def end(self):
        print('\n'.join(self.results))

if __name__ == '__main__':
    ForeachFlow()