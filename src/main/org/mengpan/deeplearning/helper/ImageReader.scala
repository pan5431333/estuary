package org.mengpan.deeplearning.helper

import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.DenseVector
import org.apache.log4j.Logger
import org.mengpan.deeplearning.helper.CatDataHelper.logger

/**
  * Created by mengpan on 2017/9/10.
  */
object ImageReader {
  val logger = Logger.getLogger(this.getClass)

  def readImageToRGBVector(fileName: String): Option[DenseVector[Double]] = {
    logger.info("Reading file: " + fileName)

    try {
      val image = ImageIO.read(new File(fileName))

      val (red, greenAndBlue) = (image.getMinX() until image.getWidth).map{
        width =>
          (image.getMinY() until image.getHeight).map{
            height =>
              val pixel = image.getRGB(width, height)
              (((pixel & 0xff0000) >> 16).toDouble, ((pixel & 0xff00) >> 8).toDouble, (pixel & 0xff).toDouble)
          }
      }
        .flatten
        .toList
        .unzip[Double, (Double, Double)](f => (f._1, (f._2, f._3)))

      val (green, blue) = greenAndBlue
        .unzip[Double, Double](asPair => (asPair._1, asPair._2))

      val flattenFeature = red ::: green ::: blue ::: Nil

      val resVector = DenseVector(flattenFeature.toArray)
      Some(resVector(200 to 400))
    } catch {
      case _: Exception => None
    }
  }

}
