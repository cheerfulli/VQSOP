from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad == None and val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))


@HOOKS.register_module()
class CommRateResetHook(Hook):
    
    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'reset_comm_rate_stats'):
            model.pts_bbox_head.reset_comm_rate_stats()
    
    def before_val_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'reset_comm_rate_stats'):
            model.pts_bbox_head.reset_comm_rate_stats()
    
    def after_val_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'pts_bbox_head') and hasattr(model.pts_bbox_head, 'get_avg_comm_rate'):
            avg_rate = model.pts_bbox_head.get_avg_comm_rate()*4*50*50*192*32/8/1024/1024
            runner.log_buffer.output['avg_comm_rate'] = avg_rate


