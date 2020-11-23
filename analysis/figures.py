import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

matplotlib.use('Agg')
sns.set_context('notebook', font_scale=1.3)
sns.set_style('white')

def str_join(args, sep='_'):
    return sep.join(map(str, args))


class BackgroundOpenNew(FileSystemEventHandler):
    """Opens newly created images"""
    def on_created(self, event):
        print(event.src_path)
        os.system(f'sleep 0.1; open -g {event.src_path}')

class Watcher(object):
    """Opens newly created files."""
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.observer = Observer()
        self.observer.schedule(BackgroundOpenNew(), self.path, recursive=True)
        self.start()

    def start(self):
        self.observer.start()

    def stop(self):
        self.observer.stop()


from datetime import datetime
class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs', hist_path='fighist', dpi=200, pdf=False):
        self.path = path
        self.hist_path = hist_path
        self.dpi = dpi
        self.pdf = pdf
        self.names = {}
        self._last = None
        self.nosave = False

        os.makedirs(path, exist_ok=True)
        os.makedirs(hist_path, exist_ok=True)

    def add_names(self, names):
        self.names.update(names)

    def nice_name(self, name):
        return self.names.get(name, name.title())

    def open(self):
        latest = max(glob(f'fighist/*'), key=os.path.getctime)
        os.system(f'open {latest}')

    def reformat_labels(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_ylabel(self.nice_name(ax.get_ylabel()))
        ax.set_xlabel(self.nice_name(ax.get_xlabel()))

    def reformat_ticks(self, yaxis=False, ax=None):
        if ax is None:
            ax = plt.gca()
        if yaxis:
            labels = [t.get_text() for t in ax.get_yticklabels()]
            new_labels = [self.nice_name(lab) for lab in labels]
            ax.set_yticklabels(new_labels)
        else:
            labels = [t.get_text() for t in ax.get_xticklabels()]
            new_labels = [self.nice_name(lab) for lab in labels]
            ax.set_xticklabels(new_labels)
        
    def reformat_legend(self, ax=None, **kws):
        if ax is None:
            ax = plt.gca()
        if ax.legend_:
            handles, labels = ax.get_legend_handles_labels()
            names = {**self.names, **kws}
            new_labels = [names.get(l, l.title()).replace('\n', ' ') for l in labels]
            ax.legend(handles=handles, labels=new_labels)

    def watch(self):
        if hasattr(self, 'watcher'):
            self.watcher.start()
        self.watcher = Watcher(self.hist_path)

    def show(self, name='tmp', pdf=None, tight=True, reformat_labels=False, reformat_legend=False, despine=True):
        if pdf is None:
            pdf = self.pdf
        try:
            if tight:
                plt.tight_layout()
            if reformat_labels:
                self.reformat_labels()
            if reformat_legend:
                self.reformat_legend()
            if despine:
                sns.despine()

            dt = datetime.now().strftime('%m-%d-%H-%M-%S')
            p = f'{dt}-{name}.png'
            tmp = f'{self.hist_path}/{p}'
            if self.nosave:
                return

            plt.savefig(tmp, dpi=self.dpi, bbox_inches='tight')

            if name != 'tmp':
                name = name.lower()
                if pdf:
                    plt.savefig(f'{self.path}/{name}.pdf', bbox_inches='tight')
                else:
                    os.system(f'cp {tmp} {self.path}/{name}.png')
        finally:
            plt.close('all')

    def figure(self, save=True, pdf=None, reformat_labels=False, reformat_legend=False, despine=True, **kwargs):
        """Decorator that calls a plotting function and saves the result."""
        def decorator(func):
            params = [v for v in kwargs.values() if v is not None]
            param_str = '_' + str_join(params).rstrip('_') if params else ''
            name = func.__name__ + param_str
            if name.startswith('plot_'):
                name = name[len('plot_'):].lower()
            try:
                plt.figure()
                func(**kwargs)
                self.show(name, pdf=pdf, reformat_labels=reformat_labels, reformat_legend=reformat_legend, despine=despine)
            finally:
                plt.close('all')
            return func
        return decorator
