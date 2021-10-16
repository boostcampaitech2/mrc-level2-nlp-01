if __name__ == "__main__":
    import k
    import os
    import sys

    print(__file__)
    root_folder = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    print(root_folder)
    sys.path.append(root_folder)

    from amo import a
else:
    from zs.train import k
    from zs import a

print(__name__)


def hello2():
    k.hello()
    a.hello3()
