package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.regularizer.Regularizer
import shapeless.{:+:, CNil, Coproduct, Generic, Inl, Inr, Lazy}

trait CanBackward[-By, -Input, +Output] {
  def backward(input: Input, by: By, regularizer: Option[Regularizer]): Output
}

object CanBackward {
  implicit val cnilCanBackward = new CanBackward[CNil, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] {
    override def backward(input: DenseMatrix[Double], by: CNil, regularizer: Option[Regularizer]) =
      throw new UnsupportedClassVersionError("Invoke backward() on CNil")
  }

  implicit def coproductCanBackward[L, R <: Coproduct](implicit lCanBackward: CanBackward[L, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])],
                                                       rCanBackward: CanBackward[R, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]) =
    new CanBackward[L :+: R, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] {
      override def backward(input: DenseMatrix[Double], by: :+:[L, R], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = by match {
        case Inl(l) => lCanBackward.backward(input, l, regularizer)
        case Inr(r) => rCanBackward.backward(input, r, regularizer)
      }
    }

  implicit def genericCanBackward[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
                                                     cCanBackward: Lazy[CanBackward[C, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]]) =
    new CanBackward[A, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] {
      override def backward(input: DenseMatrix[Double], by: A, regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) =
        cCanBackward.value.backward(input, generic.to(by), regularizer)
    }
}
