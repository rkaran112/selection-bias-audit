.PHONY: all install data run docs test clean

all: data run docs

install:
	python -m pip install -r requirements.txt

data:
	python -m sba.fetch

run:
	python run_all.py

docs:
	python make_report.py

test:
	python -m pytest tests -q

clean:
	rm -rf outputs/figures/*.png outputs/tables/*.csv \
	       outputs/headline_numbers.json docs/*.pdf slides/*.pptx
