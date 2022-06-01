# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

from pytorch3d.renderer import look_at_view_transform

#----------------------------------------------------------------------------

class CameraWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.dist          = dnnlib.EasyDict(val=1, anim=False, speed=1e-2)
        self.dist_def      = dnnlib.EasyDict(self.dist)
        self.azim         = dnnlib.EasyDict(val=0, anim=False, speed=1e-2)
        self.azim_def     = dnnlib.EasyDict(self.azim)
        self.opts           = dnnlib.EasyDict(untransform=False)
        self.opts_def       = dnnlib.EasyDict(self.opts)
        self.elev          = dnnlib.EasyDict(val=0, anim=False, speed=1e-2)
        self.elev_def      = dnnlib.EasyDict(self.elev)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Dist')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, self.dist.val = imgui.input_float('##dist', self.dist.val, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag fast##dist', width=viz.button_w)
            if dragging:
                self.dist.val += dx / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag slow##dist', width=viz.button_w)
            if dragging:
                self.dist.val += dx / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.dist.anim = imgui.checkbox('Anim##dist', self.dist.anim)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.dist.anim):
                changed, speed = imgui.slider_float('##dist_speed', self.dist.speed, -1, 1, format='Speed %.4f', power=3)
                if changed:
                    self.dist.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##dist', width=-1, enabled=(self.dist != self.dist_def)):
                self.dist = dnnlib.EasyDict(self.dist_def)

        if show:
            imgui.text('Elevation')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, self.elev.val = imgui.input_float('##elevation', self.elev.val, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag fast##elevation', width=viz.button_w)
            if dragging:
                self.elev.val += dx / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag slow##elevation', width=viz.button_w)
            if dragging:
                self.elev.val += dx / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.elev.anim = imgui.checkbox('Anim##elevation', self.elev.anim)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.elev.anim):
                changed, speed = imgui.slider_float('##elevation_speed', self.elev.speed, -1, 1, format='Speed %.4f', power=3)
                if changed:
                    self.elev.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##elevation', width=-1, enabled=(self.elev != self.elev_def)):
                self.elev = dnnlib.EasyDict(self.elev_def)

        if show:
            imgui.text('Azimuth')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _changed, self.azim.val = imgui.input_float('##azimuth', self.azim.val, format='%.4f')
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag fast##azimuth', width=viz.button_w)
            if dragging:
                self.azim.val += dx / viz.font_size * 2e-2
            imgui.same_line()
            _clicked, dragging, dx, _dy = imgui_utils.drag_button('Drag slow##azimuth', width=viz.button_w)
            if dragging:
                self.azim.val += dx / viz.font_size * 4e-4
            imgui.same_line()
            _clicked, self.azim.anim = imgui.checkbox('Anim##azimuth', self.azim.anim)
            imgui.same_line()
            with imgui_utils.item_width(-1 - viz.button_w - viz.spacing), imgui_utils.grayed_out(not self.azim.anim):
                changed, speed = imgui.slider_float('##azimuth_speed', self.azim.speed, -1, 1, format='Speed %.4f', power=3)
                if changed:
                    self.azim.speed = speed
            imgui.same_line()
            if imgui_utils.button('Reset##azimuth', width=-1, enabled=(self.azim != self.azim_def)):
                self.azim = dnnlib.EasyDict(self.azim_def)

        if show:
            imgui.set_cursor_pos_x(imgui.get_content_region_max()[0] - 1 - viz.button_w*1 - viz.font_size*16)
            _clicked, self.opts.untransform = imgui.checkbox('Untransform', self.opts.untransform)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
            if imgui_utils.button('Reset##opts', width=-1, enabled=(self.opts != self.opts_def)):
                self.opts = dnnlib.EasyDict(self.opts_def)

        """
        if self.xlate.anim:
            c = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
            t = c.copy()
            if np.max(np.abs(t)) < 1e-4:
                t += 1
            t *= 0.1 / np.hypot(*t)
            t += c[::-1] * [1, -1]
            d = t - c
            d *= (viz.frame_delta * self.xlate.speed) / np.hypot(*d)
            self.xlate.x += d[0]
            self.xlate.y += d[1]

        if self.rotate.anim:
            self.rotate.val += viz.frame_delta * self.rotate.speed
        

        pos = np.array([self.xlate.x, self.xlate.y], dtype=np.float64)
        if self.xlate.round and 'img_resolution' in viz.result:
            pos = np.rint(pos * viz.result.img_resolution) / viz.result.img_resolution

        angle = self.rotate.val * np.pi * 2
        scale = self.scale.val
        

        viz.args.input_transform = [
            [np.cos(angle) * scale,  np.sin(angle) * scale, pos[0]],
            [-np.sin(angle) * scale, np.cos(angle) * scale, pos[1]],
            [0, 0, 1]]

        """
        R, T = look_at_view_transform(dist=self.dist.val, elev=self.elev.val * 180, azim=self.azim.val * 180)
        input_transform = np.eye(4)
        input_transform[:3, :3] = R[0].numpy()
        input_transform[:3, 3] = T[0].numpy()

        viz.args.input_transform = input_transform

        viz.args.update(untransform=self.opts.untransform)

#----------------------------------------------------------------------------
