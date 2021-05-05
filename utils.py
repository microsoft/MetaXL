import logging
import fire
logger = logging.getLogger(__name__)

def calculate_parameters(module):
    return sum(p.numel() for p in module.parameters())

def print_arguments(args):
    for arg in vars(args):
        logger.info(arg, getattr(args, arg))

def get_best(file):
    lines = open(file, "r").readlines()
    best_dev = 0
    best_test = 0
    for line in lines:
        tokens = line.strip().split()
        dev = float(tokens[4])
        test = float(tokens[7])
        if dev > best_dev:
            best_dev = dev
            best_test = test
    print(tokens[0], best_dev, best_test)


def extract(in_file, out_file):
    lines = open(in_file, "r").read().split("\n\n")
    with open(out_file, "w") as f:
        for i, line in enumerate(lines[:-1]):
            print(i)
            sent = []
            for word in line.split("\n"):
                _word = word.strip().split("\t")[0].split(":")[1].strip()
                sent.append(_word)
            f.write(" ".join(sent) + "\n")

def for_latex_table():
    lines = open("haha", "r").readlines()
    lines = [str(round(float(line.strip()) * 100, 2)) for line in lines]
    print(" & ".join(lines))

def find_index(file1, file2, file3):
    import json
    index = []
    file = []
    lines1 = json.load(open(file1, "r"))
    lines2 = [line.strip() for line in open(file2, "r").readlines()]
    lines3 = [line.strip() for line in open(file3, "r").readlines()]
    for line in lines1:
        text = line["review_body"]
        if text in lines2:
            index.append(lines2.index(text))
            file.append(file2)
        elif text in lines3:
            index.append(lines3.index(text))
            file.append(file3)
        else:
            index.append(None)
    return index, file

train = json.load(open("fa.test.json", "r"))
texts = [l["review_body"] for l in train]
index = []
for text in texts:
    if text in a.text.values.tolist():
            index.append(a.text.values.tolist().index(text))
    else:
        index.append(None)
if __name__ == '__main__':
    fire.Fire()
    # for_latex_table()