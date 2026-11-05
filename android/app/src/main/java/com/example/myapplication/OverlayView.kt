package com.example.myapplication

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var dets: List<Detection> = emptyList()
    private var frameW = 1
    private var frameH = 1

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint().apply {
        color = Color.BLUE
        textSize = 40f
        isAntiAlias = true
    }

    // вызывать только с главного потока
    fun update(d: List<Detection>, w: Int, h: Int) {
        dets = d
        frameW = w
        frameH = h
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val s = min(width / frameW.toFloat(), height / frameH.toFloat())
        val dx = (width - frameW * s) / 2f
        val dy = (height - frameH * s) / 2f

        for (d in dets) {
            val b = d.box
            val l = b.left * s + dx
            val t = b.top * s + dy
            canvas.drawRect(l, t, b.right * s + dx, b.bottom * s + dy, boxPaint)
            val label = COCO_LABELS.getOrElse(d.classId) { "id${d.classId}" }
            canvas.drawText(
                "$label ${(d.score * 100).toInt()}%",
                l, (t - 6f).coerceAtLeast(textPaint.textSize), textPaint
            )
        }
    }
}