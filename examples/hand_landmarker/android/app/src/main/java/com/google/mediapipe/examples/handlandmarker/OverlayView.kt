/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: HandLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private val paint = Paint().apply {
        color = android.graphics.Color.RED
        strokeWidth = 5f
    }

    var currentMode: Int = MODE_ORIGINAL

    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        when (currentMode) {
            MODE_ORIGINAL -> {
                results?.let { handLandmarkerResult ->
                    for (landmark in handLandmarkerResult.landmarks()) {
                        for (normalizedLandmark in landmark) {
                            canvas.drawPoint(
                                normalizedLandmark.x() * imageWidth * scaleFactor,
                                normalizedLandmark.y() * imageHeight * scaleFactor,
                                pointPaint
                            )
                        }

                        HandLandmarker.HAND_CONNECTIONS.forEach {
                            canvas.drawLine(
                                landmark.get(it!!.start())
                                    .x() * imageWidth * scaleFactor,
                                landmark.get(it.start())
                                    .y() * imageHeight * scaleFactor,
                                landmark.get(it.end())
                                    .x() * imageWidth * scaleFactor,
                                landmark.get(it.end())
                                    .y() * imageHeight * scaleFactor,
                                linePaint
                            )
                        }
                    }
                }
            }
            MODE_CONVEX_HULL -> {
                results?.let { handLandmarkerResult ->
                    var normalizedLandmarks: MutableList<PointF> = ArrayList()
                    for (landmark in handLandmarkerResult.landmarks()) {
                        for (normalizedLandmark in landmark) {
                            // Scale points to canvas size
                            val scaledX = normalizedLandmark.x() * imageWidth * scaleFactor
                            val scaledY = normalizedLandmark.y() * imageHeight * scaleFactor

                            // Draw original points
                            canvas.drawPoint(scaledX, scaledY, pointPaint)

                            // Add scaled points to the list
                            normalizedLandmarks.add(PointF(scaledX, scaledY))
                        }

                        // Convert to MatOfPoint for OpenCV
                        val matOfPoint = MatOfPoint(*normalizedLandmarks.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())

                        // Calculate convex hull
                        val hull = MatOfInt()
                        Imgproc.convexHull(matOfPoint, hull)

                        // Extract hull points
                        val hullPoints = mutableListOf<PointF>()
                        val hullIndices = hull.toArray()
                        for (i in hullIndices) {
                            val point = matOfPoint.toList()[i]
                            hullPoints.add(PointF(point.x.toFloat(), point.y.toFloat()))
                        }

                        // Draw the convex hull
                        for (i in hullPoints.indices) {
                            val p1 = hullPoints[i]
                            val p2 = hullPoints[(i + 1) % hullPoints.size]
                            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, paint)
                        }
                    }
                }
            }
        }
    }

    fun setResults(
        handLandmarkerResults: HandLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE,
        resultCurrentMode: Int
    ) {
        results = handLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        currentMode = resultCurrentMode
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
        private const val MODE_ORIGINAL = 0
        private const val MODE_CONVEX_HULL = 1
    }
}
