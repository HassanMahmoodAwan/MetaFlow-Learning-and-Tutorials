from metaflow import FlowSpec, card, pypi, step, current
from metaflow.cards import Image

class FractalFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.plot)

    @pypi(python='3.9.13',
          packages={'pyfracgen': '0.0.11',
                    'matplotlib': '3.8.0'})
    @card(type='blank')
    @step
    def plot(self):
        # pylint: disable=import-error,no-member
        import pyfracgen as pf
        from matplotlib import pyplot as plt

        string = "AAAAAABBBBBB"
        xbound = (2.5, 3.4)
        ybound = (3.4, 4.0)
        res = pf.lyapunov(
            string, xbound, ybound, width=4, height=3,
            dpi=300, ninit=2000, niter=2000
        )
        pf.images.markus_lyapunov_image(res, plt.cm.bone, plt.cm.bone_r, gammas=(8, 1))
        current.card.append(Image.from_matplotlib(plt.gcf()))
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    FractalFlow()