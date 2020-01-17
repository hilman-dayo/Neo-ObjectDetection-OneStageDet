import pytest
from vedanet.engine.engine import Engine


def test_engine_instance():
    """Shows that `Engine` is an abstracted class.

    We need to inherit it first!
    """
    with pytest.raises(TypeError):
        engine = Engine()


def test_default_empty_engine_attributes():
    """Test the default attributes of our engine."""



# @pytest.mark.usefixtures("engine")
# class TestEngineDefaultAttributes:
#     """Engine Default Attributes.

#     This will show all the Engine default attributes, so that you guys will not
#     have to go against the hustle that I've went T.T"""
#     def test_batch_related_attributes(self):
#         assert engine.batch_size == 1
