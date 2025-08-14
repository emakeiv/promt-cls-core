import yaml

from prompt_cls_core.data.schema import CorpusCfg
from prompt_cls_core.data.generator import DataGenerator


def main():
    cfg_path = "src/prompt_cls_core/ops/config/corpus.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = CorpusCfg(**yaml.safe_load(f.read()))

    instance = DataGenerator()
    dataframe = instance.build_corpus(cfg)

    print(dataframe)


if __name__ == "__main__":
    main()