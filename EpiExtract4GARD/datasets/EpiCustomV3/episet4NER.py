# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""INSERT TITLE"""

import logging

import datasets


_CITATION = """\
*REDO*
@inproceedings{wang2019crossweigh,
  title={CrossWeigh: Training Named Entity Tagger from Imperfect Annotations},
  author={Wang, Zihan and Shang, Jingbo and Liu, Liyuan and Lu, Lihao and Liu, Jiacheng and Han, Jiawei},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={5157--5166},
  year={2019}
}
"""

_DESCRIPTION = """\
**REWRITE*
EpiSet4NER is a dataset generated from 620 rare disease abstracts labeled using statistical and rule-base methods. The test set was then manually corrected by a rare disease expert.
For more details see *INSERT PAPER* and https://github.com/ncats/epi4GARD/tree/master/EpiExtract4GARD#epiextract4gard
"""

_URL = "https://github.com/NCATS/epi4GARD/tree/master/EpiExtract4GARD/datasets/EpiCustomV3"
_TRAINING_FILE = "train.txt"
_VAL_FILE = "val.txt"
_TEST_FILE = "test.txt"


class EpiSetConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ConllppConfig, self).__init__(**kwargs)


class EpiSet(datasets.GeneratorBasedBuilder):
    """EpiSet4NER by GARD."""

    BUILDER_CONFIGS = [
        EpiSetConfig(name="EpiSet4NER", version=datasets.Version("1.0.0"), description="EpiSet4NER by NIH NCATS GARD"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-EPI",
                                "B-LOC",
                                "B-STAT",
                                "I-EPI",
                                "I-LOC",
                                "I-STAT",
                                "O",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/ncats/epi4GARD/tree/master/EpiExtract4GARD#epiextract4gard",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "val": f"{_URL}{_VAL_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["val"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # EpiSet tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
