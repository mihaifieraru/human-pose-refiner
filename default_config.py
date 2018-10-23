from easydict import EasyDict as edict

cfg = edict()

cfg.mean_pixel = [116.09278225718839, 105.26649931421154, 95.95121967680349, 0.34951943619606246, 0.40976568213517184, 0.49123806705952289, 0.49591614990476218, 0.41450376653672544, 0.34958344479290787, 0.43501551652668358, 0.44672932998708459, 0.54594911643432331, 0.54738354284274893, 0.45025940240059453, 0.43717204733646525, 0.4953665988006844, 0.57728900796096794, 0.48184625109052492]
cfg.std_pixel = [29.389673517428246, 29.556636640141477, 30.463665642133158, 0.29822163936637036, 0.34222439933116638, 0.41065272817038223, 0.41955244177459156, 0.33791641742457257, 0.29854783574299115, 0.47782940998262247, 0.46817435880085279, 0.46947218502800303, 0.47490794971361133, 0.46688720419242807, 0.47525211482666196, 0.47883342998379963, 0.47269364083193033, 0.45833225138386685]

cfg.global_scale = 0.8452830189
cfg.mean_diag_to_stick = 1.4286
cfg.mean_hbbox_diag = 101.4252
cfg.stride = 8.0
cfg.delta_crop = 250

cfg.weight_decay = 0.0001
cfg.pos_dist_thresh = 17

cfg.location_refinement = True
cfg.locref_stdev = 7.2801

cfg.runtime_flip = False

cfg.dir_json_pred = "json_pred/"
