package estuary.components.support

sealed trait TransformType

object TransformType {
  class FILTER_TO_COL extends TransformType
  class IMAGE_TO_COL extends TransformType
  class COL_TO_IMAGE extends TransformType
  class IMAGE_GRAD_2_COL extends TransformType
  class COL_GRAD_2_IMAGE extends TransformType
}

