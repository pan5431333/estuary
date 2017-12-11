package estuary.components.support

import breeze.linalg.DenseMatrix
import estuary.components.layers._
import shapeless.{:+:, CNil, Coproduct, Generic, Inl, Inr, Lazy}

trait CanExportParam[-From, +To] {
  def export(from: From): To
}

object CanExportParam {
  implicit val cnilCanExportParam =
    new CanExportParam[CNil, DenseMatrix[Double]] {
      override def export(from: CNil) = throw new UnsupportedClassVersionError("Invoke export() on CNil")
    }

  implicit def coproductCanExportParam[L, R <: Coproduct](implicit lCanExportParam: CanExportParam[L, DenseMatrix[Double]],
                                                          rCanExportParam: CanExportParam[R, DenseMatrix[Double]]) =
    new CanExportParam[L :+: R, DenseMatrix[Double]] {
      override def export(from: :+:[L, R]): DenseMatrix[Double] = from match {
        case Inl(l) => lCanExportParam.export(l)
        case Inr(r) => rCanExportParam.export(r)
      }
    }

  implicit def genericCanExportParam[A, C <: Coproduct](implicit generic: Generic.Aux[A, C],
                                                        cCanExportParam: CanExportParam[C, DenseMatrix[Double]]) =
    new CanExportParam[A, DenseMatrix[Double]] {
      override def export(from: A): DenseMatrix[Double] = cCanExportParam.export(generic.to(from))
    }
}

