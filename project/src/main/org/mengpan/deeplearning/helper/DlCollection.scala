package org.mengpan.deeplearning.helper

import breeze.linalg.{DenseMatrix, DenseVector, max, min}
import org.mengpan.deeplearning.data.{Cat, Data}
import breeze.stats.{mean, stddev}
import scala.collection.immutable.List

/**
  * Created by mengpan on 2017/8/15.
  */
class DlCollection[E <: Data](data: List[E]){
  private val numRows: Int = this.data.size
  private val numCols: Int = this.data.head.feature.length

  def split(trainingSize: Double): (DlCollection[E], DlCollection[E]) = {
    val splited = data.splitAt((data.length * trainingSize).toInt)

    (new DlCollection[E](splited._1), new DlCollection[E](splited._2))
  }

  def getFeatureAsMatrix: DenseMatrix[Double] = {
    val feature = DenseMatrix.zeros[Double](this.numRows, this.numCols)

    var i = 0
    this.data.foreach{eachRow =>
      feature(i, ::) := eachRow.feature.t
      i = i+1
    }

    feature
  }

  def getLabelAsVector: DenseVector[Double] = {
    val label = DenseVector.zeros[Double](this.numRows)

    var i: Int = 0
    this.data.foreach{eachRow =>
      label(i) = eachRow.label
      i += 1
    }

    label
  }


  def map[B <: Data](f: scala.Function1[E, B]): DlCollection[B] = {
    new DlCollection[B](this.data.map(f))
  }

  override def toString = s"DlCollection($numRows, $numCols, $getFeatureAsMatrix, $getLabelAsVector)"
}
