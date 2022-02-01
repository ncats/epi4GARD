# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

import csv
import os
import textwrap
import datasets
from datasets.tasks import TextClassification

_CITATION = """
John, J. N., Sid, E., & Zhu, Q. (2021). Recurrent Neural Networks to Automatically Identify Rare Disease Epidemiologic Studies from PubMed. AMIA Joint Summits on Translational Science proceedings. AMIA Joint Summits on Translational Science, 2021, 325–334.
"""

_DESCRIPTION = """\

[fix description]

Prepare positive dataset.ipynb: Generates orphanet_epi_mesh.csv, the final positive dataset (articles that are all epidemiology studies). First, PubMed IDs are extracted from a collection of epidemiology sources provided by Orphanet. The final positive set consists of the PubMed IDs that have epidemiology, incidence, or prevalence MeSH terms. The notebook includes code to optionally expand the dataset by including articles with epidemiology-related MeSH terms beyond those included in the Orphanet file, although this was shown to have worse performance.
Prepare negative dataset.ipynb: Generates negative_dataset.csv, the final negative dataset (articles that are not epidemiology studies). Using the EBI API, the top 5 PubMed search results for each of the 6,000+ rare diseases included in the GARD database are retrieved. Articles that have epidemiology MeSH terms or keywords in the abstract or that are also in the Orphanet file are removed.

negative_dataset.csv: Negative dataset assembled by Prepare negative dataset.ipynb. Columns: PubMed ID, abstract text. 25,015 rows.
orphanet_epi_mesh.csv: Positive dataset assembled by Prepare positive dataset.ipynb. Columns: PubMed ID, abstract text. 1,145 rows.
"""
_HOMEPAGE = "https://github.com/ncats/epi4GARD/tree/master#epi4gard"
_LICENSE = "https://raw.githubusercontent.com/ncats/epi4GARD/master/license.txt"

_URL = "https://huggingface.co/datasets/wzkariampuzha/EpiClassifySet/raw/main/"
_TRAINING_FILE = "epi_classify_train.tsv"
_VAL_FILE = "epi_classify_val.tsv"
_TEST_FILE = "epi_classify_test.tsv"

class EpiClassifyConfig(datasets.BuilderConfig):
    """BuilderConfig for EpiClassify."""

    def __init__(self, **kwargs):
        """BuilderConfig for EpiClassify.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(EpiClassifyConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)

class EpiClassify(datasets.GeneratorBasedBuilder):
    """The General Language Understanding Evaluation (GLUE) benchmark."""

    BUILDER_CONFIGS = [
        EpiClassifyConfig(
            name="EpiClassify",
            version=VERSION, 
            description=textwrap.dedent(
                """\
            The EpiClassify Dataset [REDO DESCRIPTION The task is to predict the sentiment of a
            given sentence. We use the two-way (positive/negative) class split, and use only
            sentence-level labels.]"""
            ),
            text_features={"abstract": "abstract"},
            label_classes=["negative", "positive"],
            label_column="label",
            #data_url="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
            #data_dir="SST-2",
        )
    ]

    def _info(self):
        #features = {text_feature: datasets.Value("string") for text_feature in self.config.text_features.keys()}
        
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(
                    names=[
                        "1 = Epi Abstract",
                        "2 = Not Epi Abstract",
                    ]
                ),
            }
        )
        
        '''
        if self.config.label_classes:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        else:
            features["label"] = datasets.Value("float32")
        features["idx"] = datasets.Value("int32")
        '''
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            task_templates=[TextClassification(text_column="text", label_column="label")],
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
    
    def _generate_examples(self, filepath, split):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as f:
            data = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
            for id_, row in enumerate(data):
                yield id_, {
                    "text": row[0],
                    "label": row[1],
                }