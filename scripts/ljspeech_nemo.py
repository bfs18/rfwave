from nemo_text_processing.text_normalization.normalize import Normalizer
from g2p_en import G2p


text_normalizer = Normalizer(input_case="cased", lang="en")
g2p = G2p()

text = "I received a present for my birthday."
normalized_text = text_normalizer.normalize(text)
print(g2p(normalized_text))
text = "The company will present a new product next month."
normalized_text = text_normalizer.normalize(text)
print(g2p(normalized_text))
