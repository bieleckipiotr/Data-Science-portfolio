import os
import matplotlib.pyplot as plt
import datasets
from PIL import Image  # Ensure correct import

class BedroomDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="LSUN Bedrooms dataset",
            features=datasets.Features({
                "image": datasets.Image(),  # Define image feature
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            ),
            
            datasets.SplitGenerator(
                name='train_subset',
                gen_kwargs={"data_dir": data_dir, 'subset': 7500},
            )
        ]

    def _generate_examples(self, data_dir, subset = None):
        index = 0
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(root, filename)
                    yield index, {
                        "image": file_path,
                    }
                    index += 1
                    if subset is not None and index >= subset:
                        return