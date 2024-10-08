"""
==================================
Figure/Axes enter and leave events
==================================

Illustrate the figure and Axes enter and leave events by changing the
frame colors on enter and leave.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event, MouseEvent


def on_enter_axes(event):
    print("enter_axes", event.inaxes)
    event.inaxes.patch.set_facecolor("yellow")
    event.canvas.draw()


def on_leave_axes(event):
    print("leave_axes", event.inaxes)
    event.inaxes.patch.set_facecolor("white")
    event.canvas.draw()


def on_enter_figure(event):
    print("enter_figure", event.canvas.figure)
    event.canvas.figure.patch.set_facecolor("red")
    event.canvas.draw()


def on_leave_figure(event):
    print("leave_figure", event.canvas.figure)
    event.canvas.figure.patch.set_facecolor("grey")
    event.canvas.draw()


fig, axs = plt.subplots(2, 1)
fig.suptitle("mouse hover over figure or Axes to trigger events")

fig.canvas.mpl_connect("figure_enter_event", on_enter_figure)
fig.canvas.mpl_connect("figure_leave_event", on_leave_figure)
fig.canvas.mpl_connect("axes_enter_event", on_enter_axes)
fig.canvas.mpl_connect("axes_leave_event", on_leave_axes)

Event("figure_enter_event", fig.canvas)._process()

MouseEvent("motion_notify_event", fig.canvas, *axs[0].transAxes.transform((0.5, 0.5)))._process()
MouseEvent("motion_notify_event", fig.canvas, *axs[1].transAxes.transform((0.5, 0.5)))._process()
MouseEvent("motion_notify_event", fig.canvas, *fig.transFigure.transform((0.1, 0.1)))._process()

Event("figure_leave_event", fig.canvas)._process()


plt.show()
