import gws.lib.os2


def install_fonts(source_dir):
    target_dir = '/usr/local/share/fonts'
    gws.lib.os2.run(['mkdir', '-p', target_dir], echo=True)
    for p in gws.lib.os2.find_files(source_dir):
        gws.lib.os2.run(['cp', '-v', p, target_dir], echo=True)

    gws.lib.os2.run(['fc-cache', '-fv'], echo=True)
