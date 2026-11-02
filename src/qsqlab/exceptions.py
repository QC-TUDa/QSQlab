# =========================== Base ============================


# ====================== PostProcessing =======================
class PostProcessingError(Exception):
    """Base class for PostProcessing errors.
    """

class MixedDataError(PostProcessingError):
    """Data contains sequences of mixed gate or measurement basis.
    """
    pass

class MixedBackendError(PostProcessingError):
    """Data contains sequences applied to mixed backend
    """
    pass

class UnevenDepthError(PostProcessingError):
    """Sequence contains an uneven depth of pi/2 rotations.
    """
    pass

class GateNotSupported(PostProcessingError):
    """Gate not supported by analysis functions.
    """
    pass

class UndefinedIdealState(PostProcessingError):
    """Sequence does not provide a well defined ideal final state.
    """
    pass

class EmptyDataFile(PostProcessingError):
    """Sequencer data file provided is empty. 
    """
    pass

# ====================== QSQPostProcessing =======================
class NoAvailableEpsilon(PostProcessingError): 
    """Analysed data has no available epsilon probability pairs.
    """
    pass