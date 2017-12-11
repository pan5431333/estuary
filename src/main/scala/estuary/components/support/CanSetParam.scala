package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._
import shapeless.{:+:, CNil, Coproduct, Generic, Inl, Inr}

trait CanSetParam[-For, From] {
  def set(from: From, foor: For): Unit
}

object CanSetParam {
  implicit val cnilCanSetParam = new CanSetParam[CNil, DenseMatrix[Double]] {
    override def set(from: DenseMatrix[Double], foor: CNil): Unit = throw new UnsupportedOperationException("Invoke set() for setting param on CNil")
  }

  implicit def coproductCanSetParam[L, R <: Coproduct](implicit lCanSetParam: CanSetParam[L, DenseMatrix[Double]],
                                                       rCanSetParam: CanSetParam[R, DenseMatrix[Double]]) =
    new CanSetParam[L :+: R, DenseMatrix[Double]] {
      override def set(from: DenseMatrix[Double], foor: :+:[L, R]): Unit = foor match {
        case Inl(l) => lCanSetParam.set(from, l)
        case Inr(r) => rCanSetParam.set(from, r)
      }
    }

  implicit def genericCanSetParam[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
                                                     cCanSetParam: CanSetParam[C, DenseMatrix[Double]]) =
    new CanSetParam[A, DenseMatrix[Double]] {
      override def set(from: DenseMatrix[Double], foor: A): Unit = cCanSetParam.set(from, generic.to(foor))
    }
}
