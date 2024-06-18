
import logging
import time
import lstm
import bi_lstm
import llm
import models
import bilstm_keras
import lstm_keras
import mbert
import tune_models
logging.basicConfig(level=logging.INFO, filemode='a', filename='models.log')


def run():
    try:
        models.main()
        lstm.main()
        bi_lstm.main()
        llm.main()
        mbert.main()
    except Exception as e:
        raise


def tune():
    try:
        bilstm_keras.run_tuning()
        lstm_keras.run_tuning()
        tune_models.run_tuning()
    except Exception as e:
        raise


def main():
    try:    
        start = time.time()
        run()
        total = time.time() - start
        logging.info("The total amount of time it took to train all the models is  {}".format(total))

        start = time.time()
        tune()
        total = time.time() - start
        logging.info("The total amount of time it took fine tune lstm and bi-lstm is  {}".format(total))
    except Exception as e:
        raise


if __name__ == "__main__":
    main()
