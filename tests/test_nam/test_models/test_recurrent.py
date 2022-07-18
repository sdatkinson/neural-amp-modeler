# File: test_recurrent.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from .base import Base

from nam.models import recurrent


class TestLSTM(Base):
    @classmethod
    def setup_class(cls):
        hidden_size = 3
        return super().setup_class(
            recurrent.LSTM,
            args=(hidden_size,),
            kwargs={"train_burn_in": 3, "train_truncate": 5, "num_layers": 2},
        )

