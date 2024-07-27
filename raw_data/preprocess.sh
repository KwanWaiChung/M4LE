if ! command -v wget &> /dev/null; then
  echo "Error: wget must be installed."
  exit 1
fi
if ! command -v gdown &> /dev/null; then
  echo "Error: gdown should be alread be installed wtih pip install."
  exit 1
fi

cd raw_data # raw_data
wget "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz" -O "wizard_of_wikipedia.tgz"
mkdir -p "wow"
tar -xzf "wizard_of_wikipedia.tgz" -C "wow"
rm "wizard_of_wikipedia.tgz"

# lost-in-the-middle data
cd ../lost_in_the_middle # lost_in_the_middle
wget https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz


# arxiv and pubmed from the paper `A discourse-aware attention model for abstractive summarization of long documents`
mkdir -p "../raw_data/summarization"
cd ../data/summarization # data/summarization
wget https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip
unzip arxiv-dataset.zip
rm arxiv-dataset.zip

wget https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip
unzip pubmed-dataset.zip
rm pubmed-dataset.zip
rm -rf __MACOSX

# news2016zh
gdown "1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp"
unzip news2016zh.zip -d news2016zh

# govdoc
gdown "1ik8uUVeIU-ky63vlnvxtfN2ZN-TUeov2"
tar -xzf "gov-report.tar.gz"
rm "gov-report.tar.gz"

# bigpatent from https://evasharma.github.io/bigpatent/
gdown "1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"
mkdir -p "bigPatentData"
tar -xzf "bigPatentData.tar.gz" -C "bigPatentData"
cd bigPatentData 
tar -xzf "train.tar.gz"
tar -xzf "test.tar.gz"
tar -xzf "val.tar.gz"
rm "train.tar.gz"
rm "test.tar.gz"
rm "val.tar.gz"
cd .. # raw_data/summarization
rm "bigPatentData.tar.gz" 

# cnn-daily
wget https://huggingface.co/datasets/cnn_dailymail/resolve/11343c3752184397d56efc19a8a7cceb68089318/data/cnn_stories.tgz
tar -xzf "cnn_stories.tgz"
rm "cnn_stories.tgz"

mkdir -p "qa/hotpotqa"
cd "qa/hotpotqa"
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# triviaQA
cd .. # raw_data/
mkdir -p "qa/triviaqa"
cd "qa/triviaqa" # raw_data/qa/trivia
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar -xzf "triviaqa-unfiltered.tar.gz"
tar -xzf "triviaqa-rc.tar.gz"
rm "triviaqa-unfiltered.tar.gz"
rm "triviaqa-rc.tar.gz"

# DRCD
cd .. # raw_data/qa
git clone https://github.com/DRCKnowledgeTeam/DRCD.git

# NQ
gsutil -m cp -R gs://natural_questions/v1.0/dev .
mv dev natural_questions

# c3
mkdir -p c3
cd c3 # raw_data/qa/c3
wget https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-m-test.json
wget https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-m-train.json
cd .. # raw_data/qa

# duorc
git clone https://github.com/duorc/duorc.git


# QMsum
cd ../summarization # data/summarization
git clone https://github.com/Yale-LILY/QMSum.git
git clone https://github.com/hahahawu/VCSum.git

# CNewsSum
gdown "1A_YcQ3cBAI7u9iVIoCeVLLgwU7UUzHHv"
unzip  CNewSum_v2.zip -d CNewSum_v2
rm -rf CNewSum_v2/__MACOSX
rm CNewSum_v2.zip

# lcsts
wget https://huggingface.co/datasets/RUCAIBox/Chinese-Generation/resolve/main/lcsts.tgz
tar -xzf "lcsts.tgz" -C "lcsts"
rm lcsts.tgz

# booksum
cd ../../booksum # booksum
gsutil cp gs://sfr-books-dataset-chapters-research/all_chapterized_books.zip .
unzip all_chapterized_books.zip
cd scripts/data_collection/bookwolf/
python get_summaries.py
cd ../cliffnotes
python get_summaries.py
cd ../gradesaver
python get_summaries.py
cd ../novelguide
python get_summaries.py
cd ../pinkmonkey
python get_summaries.py
cd ../shmoop
python get_summaries.py
cd ../sparknotes
python get_summaries.py
cd ../thebestnotes # booksum/scripts/data_collection/thebestnotes
python get_summaries.py

cd ../../data_cleaning # booksum/scripts/data_cleaning
python basic_clean.py
python split_aggregate_chaps_all_sources.py
python clean_summaries.py
cd ../../alignments/paragraph-level-summary-alignments
python gather_data.py --matched_file ../chapter-level-summary-alignments/chapter_summary_aligned_train_split.jsonl --split_paragraphs
# python align_data_bi_encoder_paraphrase.py --data_path chapter_summary_aligned_train_split.jsonl.gathered --stable_alignment


# classification
cd ../raw_data # data
mkdir -p "classification"
cd classification # data/classification
wget https://zenodo.org/record/7394851/files/MN-DS-news-classification.csv?download=1 -O "MN-DS-news-classification.csv"
mkdir -p "marc"
cd marc # data/classification/marc
wget https://huggingface.co/datasets/amazon_reviews_multi/resolve/main/json/test/dataset_en_test.json
wget https://huggingface.co/datasets/amazon_reviews_multi/resolve/main/json/test/dataset_zh_test.json
cd ..
mkdir -p "arxiv"
cd arxiv # data/classification/arxiv
wget https://huggingface.co/datasets/ccdv/arxiv-classification/resolve/main/train_data.txt
wget https://huggingface.co/datasets/ccdv/arxiv-classification/resolve/main/val_data.txt
wget https://huggingface.co/datasets/ccdv/arxiv-classification/resolve/main/test_data.txt
cd ../../raw_data


# nli
mkdir -p "nli"
cd nli # data/nli
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
rm wikitext-2-raw-v1.zip

# chinese nli
gdown "1EdHUZIDpgcBoSqbjlfNKJ3b1t0XIUjbt"
unzip wiki_zh_2019.zip
rm wiki_zh_2019.zip 
cd ..

# translation
mkdir -p "translation"
cd translation # data/translation
wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/zh_cn.zip -O zh_cn.zip
unzip zh_cn.zip
rm zh_cn.zip

# tedtalks
wget https://huggingface.co/datasets/ted_talks_iwslt/resolve/main/data/XML_releases.tgz