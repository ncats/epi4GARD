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
import csv
_CITATION = """\
*REDO*
"""
_DESCRIPTION = """\
**REWRITE*
"""
_URL = "https://huggingface.co/datasets/ncats/GARD_EpiSet4TextClassification/raw/main/"
#"https://huggingface.co/datasets/wzkariampuzha/EpiClassifySet/raw/main/"
_TRAINING_FILE = "epi_classify_train.tsv"
_VAL_FILE = "epi_classify_val.tsv"
_TEST_FILE = "epi_classify_test.tsv"
class EpiSetConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""
    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(EpiSetConfig, self).__init__(**kwargs)
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
                    "idx": datasets.Value("string"),
                    #"abstracts": datasets.Value("string"),
                    "abstracts": datasets.Sequence(datasets.Value("string")),
                    '''
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O", #(0)
                                "B-LOC", #(1)
                                "I-LOC", #(2)
                                "B-EPI", #(3)
                                "I-EPI", #(4)
                                "B-STAT", #(5)
                                "I-STAT", #(6)
                            ]
                        )
                    ),
                    '''
                    "labels": datasets.features.ClassLabel(
                        names=[
                            "1 = Epi Abstract",
                            "2 = Not Epi Abstract",
                        ]
                    ),
                    
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/ncats/epi4GARD/tree/master/Epi4GARD#epi4gard",
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
            data = csv.reader(f, delimiter="\t")
            next(data)
            for id_, row in enumerate(data):
                yield id_, {
                    "text": row[0],
                    "label": int(row[1]),
                }
        '''
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            abstracts = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n" or line == "abstract\tlabel\n":
                    if abstracts:
                        yield guid, {
                            "idx": str(guid),
                            "abstracts": abstracts,
                            "labels": labels,
                        }
                        guid += 1
                        abstracts = []
                        labels = []
                else:
                    # EpiSet abstracts are space separated
                    splits = line.split("\t")
                    abstracts.append(splits[0])
                    labels.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "idx": str(guid),
                    "abstracts": abstracts,
                    "labels": labels,
                }
                '''