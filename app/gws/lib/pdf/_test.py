"""Tests for the pdf module."""
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import pypdf

import gws
import gws.test.util as u
import gws.lib.pdf as p


def test_concat_and_page_count(tmp_path):
    a_path = tmp_path / "pdf_a.pdf"
    b_path = tmp_path / "pdf_b.pdf"
    c_path = tmp_path / "pdf_c.pdf"
    out_path = tmp_path / "output.pdf"

    a = PIL.Image.new('RGBA', (50, 50), (255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(a)
    font = PIL.ImageFont.load_default()
    draw.multiline_text((0, 0), 'A', font=font, fill=(0, 0, 0, 255))
    a = a.convert('RGB')
    a.save(a_path, format='PDF')

    b = PIL.Image.new('RGBA', (50, 50), (255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(b)
    font = PIL.ImageFont.load_default()
    draw.multiline_text((20, 20), 'B', font=font, fill=(0, 0, 0, 255))
    b = b.convert('RGB')
    b.save(b_path, format='PDF')

    c = PIL.Image.new('RGBA', (50, 50), (255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(c)
    font = PIL.ImageFont.load_default()
    draw.multiline_text((30, 30), 'C', font=font, fill=(0, 0, 0, 255))
    c = c.convert('RGB')
    c.save(c_path, format='PDF')

    p.concat([str(a_path), str(b_path), str(c_path)], str(out_path))

    assert p.page_count(str(out_path)) == 3


#testing layering order
def test_overlay(tmp_path):
    pdfA_content = b"""
    b'%PDF-1.6\n%\xc3\xa4\xc3\xbc\xc3\xb6\xc3\x9f\n2 0 obj\n<</Length 3 0 R/Filter/FlateDecode>>\nstream\nx\x9c%\xca;\n\x80@\x0c\x05\xc0>\xa7x\xb5ELV\xd7\xcd\x82X\x08\xda\x0b\x01/\xe0\x07,\x04m\xbc\xbe\x82L;\xc2\x8a\x87.\x08\x84%\x18b\x8e\x1cR\x84\xd5\xca\xd6(\xee\x95\xe6\x02\xe7?>\xf7N\xbdSl\xd8\x90R\xc5\xd92|A9*4\xc0\xb7V\xb4\xf3\x83\x06\xa7\x89&\xbc\x8e\x13\x13\xb8\nendstream\nendobj\n\n3 0 obj\n92\nendobj\n\n5 0 obj\n<</Length 6 0 R/Filter/FlateDecode/Length1 7624>>\nstream\nx\x9c\xe57mp\x1b\xc7uo\xef\x00\x12\xfc\x04HC\x8al\xd8\xc2"\'*bA\x02\xa4(\xd9\xa2,\x9a\x10I\x80\xa4H\x89\x10I\xd8\x80d\x8b8\x02G\x022\xf1a\xe0HYr<F\xda\xc8\xd6@Q\xad\xa8\x89\x13\xd9\x9a:\x9dI3\x1e\x8f;:\x98jKgT\x8b\x99$M=m\xea\xa4\xe9\x8f&\x96\x1a\xce\xd4\xe9\x9fZ#U\xb1\xd3L\\\x89}\xbbw\xa4HZ\x96\'\x9d\xfe\xeb\x92\xbb\xf7\xbe\xdf\xdb\xf7\xde\x1e\xf6\xd4\xdc\xb4\x02\xd5P\x00\x11|\xb1\x94\x9c\xb5\x13\x028\xfe\x01\x80\xd4\xc7fT\xda1\xb4\xeea\x84\x17\x00\x84\x7f\x9a\xc8N\xa6^\xfe\xeb\xc7?\x040]\x00(\xbf09ut\xe2\xf2\xd3\xd3\x8b\x00\xd5\t\x94\xd1\x12\x8a\x1c\xff\x8b\xb6\x96&\x80\xda\rh\xe3\xc1\x04\x12\xfan\x1d-G<\x88\xf8\xa6DJ}\xdaN\xa0\nq\x15q\xcbT&&\x03\xfc\x0b\x82\xb5_\xc4\xa5,%?\x9d\xad2=$ \xfee\xc4iZN)\xbf\xfd\xd3\xef\xc7\x11\xff6@U>\x9b\xc9\xabq8\x81\xfe\xeec\xf6i6\xa7d\x07_\x1e\xff!\xe2;\x00\xc4\xd3H#\xc0\xc3\xc7\x1d\x01)\xe3\xf8\xff\xf3a>\x05\xeb\xa0\xcf\xdc\x01V\xc8\xf2u\xd5\x10\xdf\x80{\xe1,\xc0\xe2\x07\x0c\xbb\xbd\xde\x1a\\\xfc\xdd\xffe\x14\x16\xfd\xf1M\xf8\x0e\\\x80S\xf0sx\xc2`\x04 \x08I\x98F\xca\xca\xf1=\xf8)R\xd9\x08\xc2\x01x\x1d\x8a\x9fb\xf6\r\x98C\xbe.\x17\x85\x17\xd9N\xee8\x82\xf0\r\x98\x85\x1f\xad\xf2\x12\x84\x14<\x83\xb1\xfc%\xfc\x9c\xb4\xc2;\xd8*\x19\xb8A,\xf0%\xf8!Z\xbd\x81\xb4\xbdw2%\xd4\xe22\xc1\xc1\x89\x15\xd4\xf7\xe0\x15\xe1$\xec\x11\xdeG\xe4,\xe3\x08^\xc1\x06?\x80s\xe4\x10ZVq\x9f\xa7\x96w\xbc\xeb\x13F_\x80gq\x1d\x81\x04\xcc \xcc\x87\xb9\xe3\xbf\x7f\x01\x15\x8b\xbf\xc6]=\x0b{\xe0\x0fa7L\xad\xd0\xb8H^\x15+\xb1~\xa3\xf0*\xe6\xf4{\x9c\xe6]b\x96\xf7\x89\x87\x85\xbf\x12\x84\x9b\x7f\x82\xc8Wa\x12\xa7Lp\xef\xc2)q\xf7\xa7d\xe8\xf7\x1eb\x08jH\xa3\xd8\x00\x15w\xe2\n\xdb\xc0z\xebw\xc2\xd6\xc5\x0f\xc5MP\t\xa1\xc5\xebK\xb4\xc5\x81\xc5_\x8b\xf2\xad\xb4i\xcct\xbf\xb9\xc3\xf4\xf7w\xf3Q\xf6US\n\xb5a\xf1W\xb7\x9e\xb9\x157\xef3\x7f\x07\xab\xf5\x1a\x80\xaf\xf7\xe0\x81H84:2\xbc?8\xb4o\xef\xe0\xc0\x9e\xfe\xbe\xde\x80\xbf\xa7\xbbk\xb7\xaf\xf3\x91\x8e]\x0f\xefl\xdf\xf1\xd0\x83\xdb[[\xbc\x9e\xe6\xa6-_\xd8\xdc\xb0I\xfa\xbc\xcb\xb9\xc1^g\xb3\xd6\xd6TUVX\xca\xcb\xcc&Q \xd0D5\x12\xf5kb\x03\xad\x0b\xc8\x92_\x92\xfb\x9a\x9b\xa8\x7fC\xa2\xa7\xb9\xc9/\x05\xa2\x1a\x95\xa9\x86\x0f\xd3f\xa9\xaf\x8f\x93$Y\xa3Q\xaam\xc6\x87\xbc\x82\x1c\xd5|(9\xb1F\xd2\xa7K\xfa\x96%\x89\x8d\xee\x82]\xcc\x85D\xb5\x1f\xf7Ht\x8e\x1c\xd8\x1fF\xf8T\x8f\x14\xa1\xdaU\x0e\xef\xe5\xb0i3Gj\x10q\xb9P\x83G\xc5\xa2\xa5~-0\x93(\xfa\xa3\x18#)UUvK\xddJes\x13\x94*\xab\x10\xacBH\xdb"eKd\xcb#\x84\x03\xc2\x16\xff\xce\x92\x00\x96\x1a\xe6\x16w\xea\x97\xe3Zp\x7f\xd8\xdf\xe3p\xb9"\xcdM\xfdZ\xad\xd4\xc3Y\xd0\xcdMje\xddZ97I\x93,t8IKM\xf3\xc5\xaf\xcc\xd9`<\xea\xae\x8eKq\xf9\xf1\xb0&\xca\xa8[\x14\xfd\xc5\xe2\x0bZ\x9d[k\x94z\xb4\xc6c\xefo\xc0\x9d+Z\x93\xd4\xe3\xd7\xdc\xcc\xea\xc0\xf0\xb2\x9f\x81\xdb.\x89fn\xb0I\xb4\xf8\x11\xe0v\xa4\xab\x1f\xac\xa6\xc8\x06\xa5\xac\xc1\xf6\x110P\x13\xba52\x1cv\xb1\xe1\x08`\xae\x8b\xc5\x80D\x03\xc5hQ\x9e[,\x8cK\xd4&\x15K\xd5\xd5\xc5\xac\x1f\xd3\r\xc10\x9a\x98[\xfc\xeeI\x87\x16\xf8JD\xb3E\x13dg\xc4\xd8z`x@\xbbg\xff\xc1\xb0&4\x04hBF\n\xfewJ\xae\x1d\x0eW\xdd\xb2L\xf0\xd3\xd8\x80i\xc1\xe4`\x86].\x96\x86\x93s>\x18GD+\xec\x0f\xeb8\x85q\xc7\x9b\xe0\xf3\xba#\x9a\x10e\x9c\xf9%\xce\xba\x10\xe3\x14\x968\xcb\xeaQ\tk;0\x12.j\xa6\x86\xfe\xb8\xe4\xc7\x8c\x9f\x94\xb5\xc28v\xd7aV\x18\xc9\xa6\xd5\xfe\xc6\xe1\x92\x8a\xf5u\xb4\xdd\x1b\xe1\xb2\x14\xa3\xea\x8f\'\xa9f\xde\x8cIB\xad\x95\n\xd87L\xa5h\xe3H\xedo\xf4\xc7U\x07:\xd8\\WO\xdb%4\xc3\xec\xf8%\x7f\xd4\xf8\x9fIl@\x03\x14\x13\xdd\xe7\xd6\x1ba4\xac\xf9z\x10\xf0\xc9F\xc5\xfc\xa5\x16/j\xc8Q,X\xb2\x87\x17S\xf3JY\xcd.u-W\x97\x85\xe5O\x8e\x84\xb9\x8a\xa1\xa6\xd9\xbb5\x88\xc6\x0c-\xcd\xeb\xe7\xe7\x8a\xfa\x8b\xd1\x1e=\x04fK\xda\x1f~\x0b\xda\x16\x17J\xdb\xa8c\xb6\r\xb6A\xa4\x87\t\xaf\xef\xc6.\xdb\xec/\x86\xe3\x13\x9a3\xea\x88\xe3\xb9\x9b\xa0a\x87K\xf3E\xb0\xc2\x11)\xacDX\xdba\x86\x1a\x17\x1c\xbc9"\xbcWF\xc3\x03#\xd2\xc0\xfe\x03\xe1\x1dF :\x83\x9935\xf8\xd7\x98\x91\xc2\x0e\xdd\x0c6\xa0fi\xb0\xd0\xb0\xe0\x10#(hC\x02\r  u\xed\xc2U+o\xb0\xe0\xb4a\xc29\x955n\xd7.\x1a&\x0eX\x92\xc60\xb4F\xeaWz\x0c9\x86\xaf2jf\xed\xd4\xdd\xb7d\xad\x8c\xa1h\xa7\xbb\xcf\xe1\x8a\xb8\xf4\xd1\xdc$ \x9b\x1a\x8eQ\xc3\xc2\x92\xda\xb7\xc4\xc2\xd7\x142,\xd8\x9f\xdd}\x9c\xc4r\xb9\x815=\rK\x8a\x14\x91\x12T\xf3\x05\xc3lo,=<\xcbF2x\xce\x8dZ\x8d\xae\xc2V$\x0b\xd3\x04.d/!,\x99Z\xc0\xedX\x99\\\xad\x97\xe3\xcbh\xdf\x1av\xff\x12\x9b\x16-\xd2\xc0H\x91\x19\x97\x0c\x83\x80\x91\xf7k\xc0Z\xd8\xb7\xa3\xce\xc1\xdf\x05\xec@K\xf8\xee\xa56<\xd2\xfc@\x17K>\x1f;\xcc\x89\x9d\xcc\x88\xd4\x1f/J#\xe1]\\\x1a\xdf\'\xcf:\x8e1_\xf50@\x06F\xbb\x9a\x9b\xf0\xd5\xd6U\x92\xc8\x89\xfd%\x1f91r \xfc\x96\r\xef\x85\'F\xc3o\nD\xe8\x8evEJ\x9b\x90\x17~\x8b\xe2\x8f\x06\xa7\n\x8c\xca\x88\x0c\xa1\x0ca\x96\x86\x11\xb1py\xc7[>\x80\x02\xe7\x9a8\x81\xe3\xb19\x02\x9cfY\xa2\x11\x88\xcd\t:\xcd\xa6;\xda\xcc\x1d\xf9@@\x8eI\xe7\xf8\x96\xa4MH\xb3\xe8\xb4\x02\xa7\xf1Q\x02\x962_\xa5\xd9g\xf1U\xf8\xaa\x85\x1a\xc1Q"\x8c\xf4&R\xbe\x8b7\xd7\n\x02\xb3\xd5\xa4\x868J\xa85\xcc\xc9s\xa4P\xaa\xf09t\x89\x02J\xf8\xf4\x08O\x84n\xbb\x0e\x1d\x08\xcfV\xe3\xaf\xb3\x83\xaf\xe8\xa8\x8b\rl\x97\r\t,6\xfe\xac\xf8i\x9c5\xca\x17#\x89b4\xc2\x0e\x1b\xac\xc7\xd2\xe0?\xd1\x88\xf4\x08\x96Iz\x04\x03)\xab\xd6*%\xa5K\xab\x92\xba\x18\xbd\x93\xd1;uz\x19\xa3\x97c\x8b\x92\xf5\x04\xd5\x0bX\xfb\xa0FX\x07\x1c\x0c\xbb\xf0H\xd2\xfb\xdeq\x14mWY\xa5"\xf8R)\xda~\xd5\x8c\xc1\xd9\xf1V3g\xee\xc3;h=\x19\xf3\xdd\xa8\xab\xb5ZM\xf55\xb6\xea\xea\xf2r\x9bI\xbc\xc7^S[W\x1b\x8d\xd4\xd7\xd5\x11\x1b\xfe>W\x97\x9b\xac\xc4:\x16\xa9$\xf5\x1f\xda\xc9\xfbv\xf2\xcfv\xf2\x03;\xb9`\'\xdf\xb6\x93\xaf\xd9\xc9\x97\xedD\xb5\x93\xb8\x9d\x8c\xdaI\x8f\x9dl\xb3\x93Mvb\xb7\x13\x93\x9d\xfc\xbe\xf2\xedwQX)m\xe22\xf3v"hv\xf2-;9m\'\x05;\xc1O\xa6\xa0\x9d\xf8\xec\xa4\xc5N\xa8\x9d\xd8\xecd\x81\x0b\xad\x11\x18\xb2\x93\'\x8c\xf1\xd4\xf2\x18{\xea\xa9\xdc\xaaq\xe8\x895\xe3\xa95\x03:\xdb\xdcu\xd0\xd6\xd6\xb6\xa1\xb3\xad\xad\xbe\xdd\xdb\xe6\x06O[]=\xf9\\{\x1d>\xda\xd9\xda\xde\xde\xda\xd2\xb0\xce\xb5\xfd!\xd2F>\xc7\x9e\xa2K$\xa2\x8b\xfc\xf8V\xef7\xc9;o\x93\xf7^\xbf\xf9\xce\x85\xe37\xaf\xbf@N\xfe;\xf9\xd9\xf6\xed\xdb\x1d\xa6\xdf~lq\xe0\x93\xfc\xd1\xadgM\x89\x9b\xd3\xec\xf2\xc5\xbe\x9c\x84{\xcf\xd2\xb6\xcb\x0f\x8fYw}\x04N\xfd\x0e\xffw=?\xf9\xc7\x95w4^Qv\xc1\x17\x0c\x02\xea\x95\xbbn\xf9\xe1\xb1e\x91\xb5\xdf`BY;\xea\xfd\x88\xf5\x03\xe8\x9f\x7f\xdc\x17\xdcc\xd8\x10\xc0\x86w\xd9\xc7\x11\xf8\xbe\xf8\xb7\xf8]\xca\xb8\x1bIz\xd9\xce\xa3\xcb6\tJ>j\xc0\x02\x94\xe3\xbd[\x87Ep\xe0\xed^\x87M(s\xc2\x80\xcdP\x83\xdf :\\\x86}\xf8\xe7\x06\\\x0e\xc7\xf0\xbbD\x87-`\'\x1e\x03\xae\x80Z\xd2e\xc0\x95$M\x82\x06\\\x05\xf7\x0bo/\x7fiz\x84_\x18p\rl\x17-\x06\\\x0b\xf7\x89\x1d,z\x13\xbb!\xbf!>f\xc0\x04\xa8I4`\x01jM\x92\x01\x8b\xf0\xa0\xa9\xd5\x80M(3i\xc0f\xb8\xcf\xf4\x82\x01\x97\xc1F\xd3\x9f\x19p9|h\xbad\xc0\x16\xd8b\x9e5\xe0\n\xb8\xdf\xfc\x9e\x01W\n\x97\xcd\xffe\xc0U\xb0\xc3\xf23\x03\xae\x86\xc7+\xaa\x0c\xb8\x06\x0eW,\xf9\xaa\x85m\x15?\xedIN&\xd5\xe41%N\xe3\xb2*\xd3X&{4\x97\x9cL\xa8tK\xac\x91nmim\xa1\xbd\x99\xcc\xe4\x94B\xbb3\xb9l&\'\xab\xc9L\xdaS\xd9\xbdVl+\x1dF\x13}\xb2\xdaD\xfb\xd31\xcf`r\\\xd1e\xe9\x88\x92KN\x0c+\x93\xd3Srnw>\xa6\xa4\xe3J\x8e6\xd3\xb5\x12k\xf1G\x95\\\x9e![=\xad\x9e\xed\xb7\x99ke\x93y\xbc\xb5\xab99\xae\xa4\xe4\xdc\x9343\xb1:\x0e\x9aS&\x93yU\xc9!1\x99\xa6!\xcf\x88\x87\x06eUI\xabTN\xc7\xe9\xe8\xb2\xe2\xd0\xc4D2\xa6pbL\xc9\xa92\ng\xd4\x04Fzx:\x97\xcc\xc7\x931\xe6-\xefY\xde\xc0\x8al\x8c\xa8\xca\x8cB\xf7\xca\xaa\xaa\xe43\xe9.9\x8f\xbe0\xb2\xd1d:\x93o\xa2G\x12\xc9X\x82\x1e\x91\xf34\xae\xe4\x93\x93id\x8e\x1f\xa5\xabu(re\xdcK:\x9d\x99A\x933J\x13\xc6=\x91S\xf2\x89dz\x92\xe6\xd9\x96\rm\xaa&d\x95m:\xa5\xa8\xb9dL\x9e\x9a:\x8a%KeQk\x1ckt$\xa9&\xd0qJ\xc9\xd3}\xca\x11:\x9cI\xc9\xe9\xd7=z(\x98\x9b\t\xcc)M\xa6\xb2\xb9\xcc\x0c\x8f\xb19\x1f\xcb)J\x1a\x9d\xc9qy<9\x95T\xd1ZB\xce\xc91\xcc\x18\xa6-\x19\xcb\xf3\x8c`"hVN7\xfb\xa7s\x99\xac\x82\x91>\xd6;x[\x10\x03\xd4\xb3\x99\xcfL\xcd\xa0g&\x9dV\x948\xf3\x88a\xcf(S\xa8\x84\x8e\xa72\x99\'\xd9~&29\x0c4\xae&\x9aWD>\x91I\xab\xa8\x9a\xa1r<\x8e\x1b\xc7leb\xd3)V\'L\xb3\xba\x14\x9c\x1c\xcbe\x90\x97\x9d\x92U\xb4\x92\xca{\x12\xaa\x9a\xdd\xe9\xf5\x1e9r\xc4#\x1b\xa5\x89ae<h\xd9{7\x9ez4\xab\x18\xf5\xc81+\xa9\xa9A,\x7f\x9a\x95n\x9a\xd7\x97mb\xa4\x7f\x90\x0ee1?\x01\x0c\x8e\x1a\x02Mt\xa93[=\xad\x86\x0bLc2\xab\xe6=\xf9\xe4\x94\'\x93\x9b\xf4\x0e\x05\x06\xa1\x07\x92\xf8)\x9e\x04\x15\xe71P \x0e\x14\xa7\x8c\xb8\x8cP\x0c2\x90\x85\xa3\x90\xe3R\t\xa4R\xd8\x82\xd4F|n\x85\x16h\xc5I\xa1\x17\xa52\xc8\x9fB}\n\xdd\x08\xe7P\x8b\xad2\xb7\x9b\x814x\xf0\x93\xb9\xfb3\xadmEh\xd8\x88\xa2\x8fk7!\xd4\x8f\xfa1\xb40\x88z\xe3\xc8]i\x97\xc2\x08\xa7$\xf15\xcb4\'a\x1a\xe3\x90\x91\xb2\x1b\xf2\xa8\xa5\xa0L\x9cKPh\xc6\xf9Y6>\x8b\xff(\x87\xf2\xcb\x9c\xad\x18W+\xce\xedw\xd4\xfc,\xbbI\xb4Dy\xa6U\xcea\x91\xa6x\xf4O"-\x83zw\xcb\x07E9\x85W/\x8f\x1c\x85cqn\x95\xd9\x0e\xa1\xc4\x08\x97\nrM\x96\t\x95{Ks\xa9\xd1;x\x1cB\x8f\x13\xa8\x1f\xe3\x95\\\x92\x8cq\xdb\xac#t\xcb\x19\x84\x13FN\x0fc\xbes<\x828\xd7[\xda[\x1e=\x7f\xb2\x02w\xee\x8d\x11\x1e\xdd\x0c\xf7\xb9\x97\xd3\x19\x9e\xe7\xbc.\xc4\xf3\xc6\xbe\xf4\x9c\x8d\xf2(2He\xb98\x82\x910\xbf\t\x0e\xcb<\x9fq\xae\xcdz,mh\x8ec\xd7\xd1\xbb\xfa\xa1\x86\xael\xd4%\xcd}\xcc\x18Q2\x9d&#\xdf\x13|\xcds\xbfi\xf4Ay|z\x95W\xfb\xa6<O2\xcf\xba^\xe9\x14rU.\x1bC\xfa\x14\xfe\x1d5NY\n\xb3\xa2\xfb\x1a7\xce\xd1\x11~*\x13\xc6\x8eS\xdc.\x85}\xf8<\xc2\xbb"\xc3\xeb\x96v}\x9e\xd7\xf8vV\xf4\xbe\x990\xfa\x94r\xdd,\xc2\x19\xbe\x8b\xa5<6\xf3\xda\xb0\x9d(<R\x06\xc9\xfc\xe4\x8f\xa3\xc6\x14\xf7\xad\xc7\x96\xe0\xdd!\xf3\xda*F\xadU\xbe\x83\xa5|\xc5\x8d\x9d\xb2\xa8\xb3\x9c\xd2\x0c~\xde\x17\xec\xbc+FN\x1f\xc3\xf7\xc4\xe0\x1d-\xea\x19\\\xd9\x9b\xac&S<\xde\xfc\n\xdbi\x1em|y\x8fz\xb6\x99\xd4\x94\xe1I\xdf\xf1\x14\x7f\x1f=\xb9\\\x9f\t\xdeozF\xe3\xdcZ\xf3\xa7\xe4|\x82\xe7F5\xbcfxDq\xfc\xd3+\xae\xf7V\x06u\xa7y=\xf4\xf3\xa4w\xb3\xfa\x89\xcc\xc9<\xbf\x19C/\xcb\xdfJ\xaa\x11K\x8a\x9f\x8f\x04\xef\xc0,\xec\xc4\x8b\xa5\x17\xa3c\x7f\x1e\xde\x87+OM\xcc83\x1e#f\xef\xffZ\x8f\xc5\x95\xe5\x19\\y>r\xcb\xb1\xa40\xc6A\xe3\xf4\xa7\x97O\xdd\xf4\x8a\xf3\xbbT\x89\x11|\x07\r\xf2\xf7E\xd6\xe8\x9f\x80\x919\xba\xc6\x02;5k\xdf\x99\xad\xfc\x9d\xb9z\x17z7&\x11Wy<y\x9eK\x0f\xdf\xc3$\xf2\x87\xd0\xc3 \xbbC\xf3\xb1x\x1cC\xba\xc3(U\x04w\x8f\x13\x05\x08I\x90I\xbc\xba;I\x14\xf6\x911\x08\x91\xdd\xd0A|\xf8\xc4\x8fd\xbc<\x87H7\xe2\xec\xe9!\x1dP@\xb9\x0e\xa4?\x82\xf8.\xa4?\x8c\xefN\'\xae\x9d8\x87p\xbe\x88\xd3\x84S\x97hA\t/>\xbd\x06\xde\x8cx\x13j\xbc\x8b+\xe1\x93Q;\x91\xca\x9e{\x10\xef\xc3g\xaf\xf1\x0c \xdd\x8fO\xbf\x81\xf7#\x8eO\x88\x92r\xbc\x84w\xf2\xf5\x121\xf9f\xc9\xc2M\xf2\xeeMBo\x92\xe7>&\xc1\x8fI\xe1\xc6\xe9\x1b\xc2\x7f^ot\x9e\xbf~\xe9\xba0tm\xec\xda\xf9kb\xcb5b\xbdF,p\xd5v5x5z5{\xf5[W\xcb*\xad\x1f\x90j\xf8\x0fR\xf7o\x0b;\x9c\xbf\xec\xb8\x12\xfa\xd7\x8e\xcb!\xb8\x82;\xbb\xd2r%x\xa5pE\xbbb\xbeB\xc4\xd0eq\xbd\xd36O\xe7[\xe6\xb3\xf3\x85\xf9\x9f\xcc/\xcc_\x9f\xb7\x14\xde>\xfd\xb6\xf07\x17\xbdN\xebE\xe7E\xc19;4\xfb\xdc\xac\x18}\x8dX_s\xbe&\x04_\x89\xbe"\x9c>G\xac\xe7\x9c\xe7\xbc\xe7\xc4\x97\xcfz\x9cg{7:\xbf\xf1\xd2\x17\x9c\x0b/]\x7fI\x98[\x9c\x9f}\xa9\xa6.p\x91\x0c\x91A\xe8\xc0\x1c\xee\x9b\x15\x17\x9d\xe7w\xaf#{q[V\\\x9d8\xbd8\x87pfp\xbe\x88\x13\xbfyP\xdc\x89\xd3K\x06};\xc4\xb1\xaf\x93\xaa3\x8e3\xee3\xcf\x9c9y\xc6\x9c}\xbe\xf0\xfc\xe9\xe7\xc5\xc2\xf1\xd3\xc7\x85\xf33\x97f\x84|\xb0\xd1\x99I\xbb\x9d\xe9\xde?p\xde\xdb\xb6!T\xde&\x86\xca\xd0\rz\xf7\xf5\x8f7l\tD\xc7|\xce1\x14:x\xa0\xc5y\xa0\xb7\xd1yO[}\xc8\x8c\x1b6\xa1\xa0Ut\x8a\x9d\xe2\x90\x98\x11_\x14/\x89\xe5\x96\xe1\xe0F\xe7~\x9c\x0b\xc1\xebA\xc1\x17\xac\xa8\x0eX\x87\x9cC\xde!qnq\xc1\xa7\x0c\xb8\xd0\xda\x9e\xec\x9e\xc2\x1e\xb1?\xd0\xe8\xec\xeb\xdd\xe1\xb4\xf6:{\xbd\xbd\xef\xf6\xfe\xb2\xf7Zo\xd9X/y\x15\xff\x03\xe7\x03\x97\x02\xa2/\xd0\xe8\r\xf8\x02\x1b]\x81\xfb\xfb\x1c\xa1\xf5m\xebBu\xc4\x1a\xb2\xb5YC\x02\xc1B\xb7A\xc8k]\xb4\nV\xeb\x98\xf59\xabh\x85N\x10\n\xeb\x89\x99\xcc\x91\xd3\xa5\xd1\x11\xb7{`\xae|qx@\xb3\x04\x0fj\xe4\x84\xd60\xc2V\xdf\xfe\x03Z\xd9\t\rB\x07\x0e\x86K\x84\xfcq\xe4\xf8\xa9S\xd0\xf5\xc0\x80\xb6u$\xacE\x1f\x88\x0chq\x04|\x0c( `{\xa0\xb4\x1e\xba"\xf9\xbc\xea\xe6\x83\xb8\xdd\x08O\xe3\n\xeei7\x12\x0f\xe5u*,\xf3\xc1\x9d\'y|E\xe5\xb9\x12q3\x01\x1d\'\xb8\xba\x19\x0f\tL\x8f\xa0\xf6\xa1<\xb0\x851\xdd\xba\x12\xd3\xce\x1b\xe6\xb8\xb2\xbep`\xc3\xa1\xff\x01$\xd2\xa3\xef\nendstream\nendobj\n\n6 0 obj\n4141\nendobj\n\n7 0 obj\n<</Type/FontDescriptor/FontName/BAAAAA+LiberationSerif\n/Flags 4\n/FontBBox[-543 -303 1277 981]/ItalicAngle 0\n/Ascent 0\n/Descent 0\n/CapHeight 981\n/StemV 80\n/FontFile2 5 0 R\n>>\nendobj\n\n8 0 obj\n<</Length 222/Filter/FlateDecode>>\nstream\nx\x9c]\x90Ak\xc4 \x10\x85\xef\xfe\x8a9\xee\x1e\x16\xcd\xd2c\x08\x94-\x0b9t[\x9a\xf6\x07\x18\x9dd\x85f\x94\x899\xe4\xdfwb\xd3\x16zPx\xbe\xf7\xe9\x1b\xf5\xa5}j)d\xfd\xca\xd1u\x98a\x08\xe4\x19\xe7\xb8\xb0C\xe8q\x0c\xa4\xaa3\xf8\xe0\xf2\xae\xca\xee&\x9b\x94\x16\xb6[\xe7\x8cSKC\xack\xa5\xdf\xc4\x9b3\xafpx\xf4\xb1\xc7\xa3\xd2/\xec\x91\x03\x8dp\xf8\xb8t\xa2\xbb%\xa5O\x9c\x902\x18\xd54\xe0q\x90{\x9em\xba\xd9\tu\xa1N\xad\x17;\xe4\xf5$\xc8_\xe0}M\x08\xe7\xa2\xab\xef*.z\x9c\x93u\xc8\x96FT\xb51\r\xd4\xd7k\xa3\x90\xfc?o\'\xfa\xc1\xdd-K\xb2\x92\xa41\x0fU\xc9\xee\xa7\x1b\xb5\x8d\xf5\xd3\x06\xdc\xc2,M\xca\xec\xa5\xc2\xf6x \xfc\xfd\x9e\x14\xd3F\x95\xf5\x05~<mx\nendstream\nendobj\n\n9 0 obj\n<</Type/Font/Subtype/TrueType/BaseFont/BAAAAA+LiberationSerif\n/FirstChar 0\n/LastChar 1\n/Widths[777 722 ]\n/FontDescriptor 7 0 R\n/ToUnicode 8 0 R\n>>\nendobj\n\n10 0 obj\n<</F1 9 0 R\n>>\nendobj\n\n11 0 obj\n<</Font 10 0 R\n/ProcSet[/PDF/Text]\n>>\nendobj\n\n1 0 obj\n<</Type/Page/Parent 4 0 R/Resources 11 0 R/MediaBox[0 0 595.303937007874 841.889763779528]/Group<</S/Transparency/CS/DeviceRGB/I true>>/Contents 2 0 R>>\nendobj\n\n4 0 obj\n<</Type/Pages\n/Resources 11 0 R\n/MediaBox[ 0 0 595.303937007874 841.889763779528 ]\n/Kids[ 1 0 R ]\n/Count 1>>\nendobj\n\n12 0 obj\n<</Type/Catalog/Pages 4 0 R\n/OpenAction[1 0 R /XYZ null null 0]\n/Lang(de-DE)\n>>\nendobj\n\n13 0 obj\n<</Creator<FEFF005700720069007400650072>\n/Producer<FEFF004C0069006200720065004F0066006600690063006500200037002E0033>\n/CreationDate(D:20240504130528+02\'00\')>>\nendobj\n\nxref\n0 14\n0000000000 65535 f \n0000005178 00000 n \n0000000019 00000 n \n0000000182 00000 n \n0000005347 00000 n \n0000000201 00000 n \n0000004426 00000 n \n0000004447 00000 n \n0000004637 00000 n \n0000004928 00000 n \n0000005091 00000 n \n0000005123 00000 n \n0000005472 00000 n \n0000005569 00000 n \ntrailer\n<</Size 14/Root 12 0 R\n/Info 13 0 R\n/ID [ <BE55FC033E09703C9E8EDA2798B3121F>\n<BE55FC033E09703C9E8EDA2798B3121F> ]\n/DocChecksum /6E166EA57084D948E6CBE917E18474E0\n>>\nstartxref\n5744\n%%EOF\n'
    """

    pdfB_content = b"""
    b'%PDF-1.6\n%\xc3\xa4\xc3\xbc\xc3\xb6\xc3\x9f\n2 0 obj\n<</Length 3 0 R/Filter/FlateDecode>>\nstream\nx\x9c%\xca;\n\x80@\x0c\x05\xc0>\xa7x\xb5ELV\xd7\xcd\x82X\x08\xda\x0b\x01/\xe0\x07,\x04m\xbc\xbe\x82L;\xc2\x8a\x87.\x08\x84%\x18b\x8e\x1cR\x84\xd5\xca\xd6(\xee\x95\xe6\x02\xe7?>\xf7N\xbdSl\xd8\x90R\xc5\xd92|A9*4\xc0\xb7V\xb4\xf3\x83\x06\xa7\x89&\xbc\x8e\x13\x13\xb8\nendstream\nendobj\n\n3 0 obj\n92\nendobj\n\n5 0 obj\n<</Length 6 0 R/Filter/FlateDecode/Length1 7540>>\nstream\nx\x9c\xe57mp[Uv\xe7\xbe\'\xd9\xf2\xa7\xe4$v\r\n\xd1\x15\x0f\x07\xb9\xb2%;\x1fl\xbe\x8c\x15\xdb\x92\xed\xd8\x89\x15\xdb\x02)\x81X\xcf\xd2\xb3\xa5`} =;\x9bP\x06mw\x80\x8c\xb2iBv\n\rd\x17\xb6\xd32\x0ce7\xcf\x98m\xcdN\x96\x98\xd9\xaf\xeet\xdb\xb0\xed\x8f\xce.\xa4d\xa6\xcb\xaf\x92I\x9a\x85\xed\x0eK\xec\x9e{\xdf\x93\xe3\x84\x00\xd3N\xff\xf5\xc9\xf7\xbe\xf3}\xce=\xe7\xdc\xeb\xfb\xd4\xdc\xb4\x02\xd5P\x00\x11|\xb1\x94\x9c]C\x08\xe0\xf3\x0b\x00\xb2*6\xa3\xd2\x8e\xa1\xfa\xed\x08_\x02\x10\xfey";\x99z\xfe\xef\x1e\xfa\x08\xc0\xf4\x06@\xf9\x1b\x93S\x87\'ND~\xf8\x14@u\x02e\x1eJ(r\xfc[\x1b\x0e\xba\x01j~\x8c6\xeeK \xa1o\xf1p9\xe2\x9f"~O"\xa5~u;\x9c\xa8\x02\xa8\xa5\x88[\xa621\x19\xe0{\x08\xd6\xbap*K\xc9_\xcd:L[\x04\xc4\xdb\x10\xa7i9\xa5\xfc\xfe\xdb?\x8a#\x1e\x04\xa8\xcag3y5\x0eG\x97\x00\xee`\xf6i6\xa7d\x07\x9f\x1f\xff\t\xe2\xbf\x01\x10O"\x8d\x00\x0f\x1fW\x04\xa4\x8c\xe3\xff\xcf\x1f\xf3q\xa8\x87>s\x07X!\xcb\xe7\x9b\x1e\xf15\xb8\x03N\x03,}\xc8\xb0\x1b\xf3\xe2\xe0\xd2\'\xff\x97QX\xf4\xd7_\xc0\xcb\xf0\x06\x1c\x87_\xc1\xc3\x06#\x00AH\xc24RV>o\xc3/\x91\xca\x9e \xec\x83W\xa1\xf89f_\x83y\xe4\xebrQ8\xc1Vr\xdb\'\x08\xcf\xc1\x1c\xfc\xec&/AH\xc1c\x18\xcb\xf7\xe1W\xa4\x1d~\x8e\xad\x92\x81k\xc4\x02_\x83\x9f\xa0\xd5kH\xdb};SB-N\x13\x1c\x9cXA}\x17^\x10\x8e\xc1.\x01\xfb\x10\xa3@\x8e\xe0\x15l\xf0c8C\x0e\xa0e\x15\xd7y|y\xc5;>c\xf4ix\x1c\xe7\x11H\xc0\x0c\xc2\xfc1w|\xfak\xa8X\xfa-\xae\xeaq\xd8\x05\x7f\n;aj\x85\xc69\xf2\xa2X\x89\xf5\x1b\x85\x171\xa7os\x9a\xb7\xc4,\xef\x13\x0f\n\x7f+\x08\xd7\xbf\x89\xc830\x89C&\xb8v\xe1\xb8\xb8\xf3s2\xf4?~\xc4\x10\xd4\x90f\xb1\t*n\xc7\x156\x81u\xf1\x13a\xc3\xd2G\xe2=P\t\xa1\xa5\xab%\xda\xd2\xc0\xd2oEy1m\x1a3\xad5w\x98\xfe\xe1\x8b|\x94=cJ\xa16,}\xb0\xf8\xd8b\xdc\xbc\xc7\xfc2V\xeb\x15\x00_\xef\xfe}\x91phtdxoph\xcf\xee\xc1\x81]\xfd}\xbd\x01\x7fOw\xd7N_\xe7\xfd\x1d;\xb6o\xdb\xba\xe5+\xf7mno\xf3zZ[\\\xf7\xaeo\xbaG\xba\xdb\xe9h\\Sg\xb3\xd6\xd6TUVX\xca\xcb\xcc&Q \xd0B5\x12\xf5kb\x13\xad\x0b\xc8\x92_\x92\xfbZ[\xa8\xbf1\xd1\xd3\xda\xe2\x97\x02Q\x8d\xcaT\xc3\x97i\xbd\xd4\xd7\xc7I\x92\xac\xd1(\xd5\xd6\xe3K^A\x8ej>\x94\x9c\xb8E\xd2\xa7K\xfa\x96%\x89\x8d\xee\x80\x1d\xcc\x85D\xb5\x7f\xec\x91\xe8<\xd9\xb77\x8c\xf0\xf1\x1e)B\xb5\xcb\x1c\xde\xcda\xd3z\x8e\xd4 \xe2t\xa2\x06\x8f\x8aEK\xfdZ`&Q\xf4G1F2[U\xd9-u+\x95\xad-0[Y\x85`\x15B\x9aK\xca\xce\x12\xd7\xfd\x84\x03\x82\xcb\xbfmV\x00K\rs\x8b+\xf5\xcbq-\xb87\xec\xef\xb1;\x9d\x91\xd6\x96~\xadV\xea\xe1,\xe8\xe6&\xb5\xb2n\xad\x9c\x9b\xa4I\x16:\x1c\xa3\xb3-\x0b\xc5o\xcc\xdb`<\xea\xae\x8eKq\xf9\xa1\xb0&\xca\xa8[\x14\xfd\xc5\xe2\xd3Z\x9d[k\x96z\xb4\xe6#\xbfi\xc4\x95+Z\x8b\xd4\xe3\xd7\xdc\xcc\xea\xc0\xf0\xb2\x9f\x81\x1b.\x89fn\xb2I\xb4\xf81\xe0r\xa4\xcb\x1f\xdeL\x91\rJY\x93\xedc`\xa0&tkd8\xecd\x8f=\x80\xb9.\x16\x03\x12\r\x14\xa3Ey~\xa90.Q\x9bT\x9c\xad\xae.f\xfd\x98n\x08\x86\xd1\xc4\xfc\xd2\x0f\x8e\xd9\xb5\xc07"\x9a-\x9a \xdb"\xc6\xd2\x03\xc3\x03\xda\xea\xbd\xfb\xc3\x9a\xd0\x14\xa0\t\x19)\xf8\xd7)9\xb7\xd8\x9du\xcb2\xc1\xcfc\x03\xa6\x05\x93\x83\x19v:Y\x1a\x8e\xcd\xfb`\x1c\x11\xad\xb07\xac\xe3\x14\xc6\xed\xaf\x83\xcf\xeb\x8ehB\x94q\x16J\x9c\xfa\x10\xe3\x14J\x9ce\xf5\xa8\x84\xb5\x1d\x18\t\x175SS\x7f\\\xf2c\xc6\x8f\xc9Za\x1c\xbb\xeb +\x8cd\xd3j\x7fgwJ\xc5Uut\xab7\xc2e)F\xd5\x1fOR\xcd\xbc\x1e\x93\x84Z+\x15\xb0o\x98J\xd1\xc6\x91\xda\xdf\xe9\xaf\xcbvt\xb0\xben\x15\xdd*\xa1\x19f\xc7/\xf9\xa3\xc6\xdfL\xa2\x11\rPLt\x9f[o\x84\xd1\xb0\xe6\xebA\xc0\'\x1b\x15\xf3\xcf\xb6yQC\x8eb\xc1\x92=\xbc\x98\x9aW\xcajk\xa4\xae\xe5\xea\xb2\xb0\xfc\xc9\x910W1\xd4\xb45\xdd\x1aDc\x86\x96\xe6\xf5\xf3}E\xfd\xc5h\x8f\x1e\x02\xb3%\xed\r\xbf\t\x1b\x97.\xcdn\xa2\xf6\xb9\x8d\xb0\t"=L\xb8\xa1\x1b\xbbl\xbd\xbf\x18\x8eOh\x8e\xa8=\x8e\xfbn\x82\x86\xedN\xcd\x17\xc1\nG\xa4\xb0\x12am\x87\x19j\xbed\xe7\xcd\x11\xe1\xbd2\x1a\x1e\x18\x91\x06\xf6\xee\x0bo1\x02\xd1\x19\xcc\x9c\xa9\xc9\x7f\x8b\x19)l\xd7\xcd`\x03j\x96&\x0b\r\x0bv1\x82\x826$\xd0\x00\x02R\xd7\x0e\x9c\xb5\xf2&\x0b\x0e\x1b&\x9cSY\xe3v\xed\xa0ab\x87\x924\x86\xa15S\xbf\xd2c\xc81\xfc&\xa3f\xd6N\xdd}%ke\x0cE;\xdd}vg\xc4\xa9?\xad-\x02\xb2\xa9\xe1\x185,,\xa9}%\x16\x1eS\xc8\xb0`\x7fv\xf7q\x12\xcbe#kz\x1a\x96\x14)"%\xa8\xe6\x0b\x86\xd9\xdaXzx\x96\x8dd\xf0\x9c\x1b\xb5\x1a\xbd\t[\x91,L\x138\x91]BX2\xb5\x80\xdb\xbe2\xb9Z/\xc7\x97\xd1\xbe[\xd8\xfd%6-Z\xa4\x81\x91"3.\x19\x06\x01#\xef\xd7\x80\xb5\xb0oK\x9d\x9d\x9f\x05lCKx\xf6R\x1bni\xbe\xa1\x8b\xb3>\x1f\xdb\xcc\x89m\xcc\x88\xd4\x1f/J#\xe1\x1d\\\x1a\xcf\x93\xc7\xedG\x98\xafU0@\x06F\xbbZ[\xf0h\xeb\x9a\x95\xc8\xd1\xbd\xb3>rtd_\xf8M\x1b\xde\x0b\x8f\x8e\x86_\x17\x88\xd0\x1d\xed\x8a\xcc\xde\x83\xbc\xf0\x9b\x14\xffip\xaa\xc0\xa8\x8c\xc8\x10\xca\x10fi\x18\x11\x0b\x97\xb7\xbf\xe9\x03(p\xae\x89\x138\x1e\x9b\'\xc0i\x96\x12\x8d@l^\xd0i6\xdd\xd1z\xee\xc8\x07\x02rL:\xc7W\x926!\xcd\xa2\xd3\n\x9c\xc6\x9fY`)\xf3U\x9a}\x16_\x85\xafZ\xa8\x11\xec\xb3\x84\x91^G\xca\x0f\xf0\xe6ZA`\xae\x9a\xd4\x10\xfb,j\rs\xf2<)\xccV\xf8\xec\xbaD\x01%|z\x84GC7\\\x87\xf6\x85\xe7\xaa\xf1\xbf\xb3\x9d\xcf\xe8\xa8\x8b=\xd8.\x8d\t,6\xfe[\xf1\xd38k\x94?\x89$\x8a\xd1\x08\xdbl\xd0\x80\xa5\xc1?\xa2\x11\xe9~,\x93t?\x06RV\xadUJJ\x97V%u1z\'\xa3w\xea\xf42F/\xc7\x16%\r\x04\xd5\x0bX\xfb\xa0FX\x07\xec\x0f;qK\xd2;\x7fn/\xda.\xb3JE\xf0P)\xda>h\xc5\xef\x8b\xedK\x9f\x98\xae\xe1\x1d\xd4\x02\xab`=\\\xf4}\xb3\xe2nXk\xae\xad\xad\xafw\xac\xbd\xdb\xe4\xba\xb7\xc9\x16\x8d4\xadr\x98\xab\xcd\xd5\xd1\x88\xd5L\xaaD\xb3\xb9quc\xc3X\xa4\xd1\x14\x8d4\x8a\xab\xeb\xc7"\xabW\xbd\xe4"\']\xa4\xe0"Y\x17\x89\xba\x88\xcfE.\xb9\xc8\x8b\x9c\x82h\x90S\xa8\x8b\xbc\xef"\x0b\x9c\xd2\xc6Qp\x91m\x178\xdb\xe6"W\xb9\t\xe0\x9a\xef\xb8\xc8K\xdc\x96\xae\xf90\x7f\x1e}\xf4\xd1\\.w`\x19yT\'\xe0\x03\x9dnh\xect\xbb\xebV\xc1\xd6F\xef\xd8\x81\x87u\xa8\x13_\xe4\x8f\xb6\xd6m\xd4\x7f\xedmd\xd3z7\xa9\xdb\xb8\xe1\xbe\xd5\x1ch@\xc8\xbc\xf9+u\xf7nv\xd2\x86\xfa5e\xe5\xebH\xfd\x1a\x93\xb3I|\xeb\xf9\xef^x\xf7/O+\xe7.\\-\x9e\xf9\x9bW?m|\xed5A\xc1+\xf83_\xff\xfeO\x17?^\x82\xc5Q\xf1\xf7\x8fe\x17\xcd\x85\xc5\x86\xe3_\xbf\xfe\x8b\xb2g>\xd8l7}\xfb\xce\xcd\xcf\xfd\xd5\xcc\xcbkW\x7f\xf7\xe8\xdb?+}\xe5\x08w\x9c>\x9b\xad\xff\xd71\xeb\x8e\x8f\xc1\xa1\xdf\xb7\xff\xbe\xe7\x9d\x7f\xbaq\x9b2\xb2\xcf.\xe3\x82AB\xbdr\xe7\xa2\x1f\x1e\\\x16\xba\xf5{I(\xdb\x8aw\xd1\x07`;\xc7\xbeW\xf2\x05\xf7b=9\x1flx\xef|\x08\x81\x1f\x89?E\x1a\xe3\xae#\xe9e;\x0f,\xdb$(\xf9\x80\x01\x0bP\x8ewd\x1d\x16\xc1\x8e7q\x1d6\xa1\xccQ\x036C\r~/\xe8p\x19~\xb7\xfc\xb5\x01\x97\xc3\x11\xfc\x86\xd0a\x0b\xac!\x1e\x03\xae\x80Z\xd2e\xc0\x95$M\x82\x06\\\x05k\x85\xb7\x96\xbf\n=\xc2\xaf\r\xb8\x066\x8b\x16\x03\xae\x85;\xc5\x0e\x16\xbd\x89\xddf_\x13\x1f4`\x02\xd4$\x1a\xb0\x00\xb5&\xc9\x80E\xb8\xcf\xd4n\xc0&\x94\x994`3\xdciz\xda\x80\xcb`\x9d\xe9;\x06\\\x0e\x1f\x99\xce\x1b\xb0\x05\\\xe69\x03\xae\xc0\xfe\x7f\xd7\x80+\x85\xf7\xcc\xffe\xc0U\xb0\xc5\xf2/\x06\\\r\x0fUT\x19p\r\x1c\xac(\xf9\xaa\x85M\x15\xbf\xecIN&\xd5\xe4\x11%N\xe3\xb2*\xd3X&{8\x97\x9cL\xa8\xd4\x15k\xa6\x1b\xda\xda\xdbho&39\xa5\xd0\xeeL.\x9b\xc9\xc9j2\x93\xf6Tv\xdf*\xb6\x81\x0e\xa3\x89>Ym\xa1\xfd\xe9\x98g09\xae\xe8\xb2tD\xc9%\'\x86\x95\xc9\xe9)9\xb73\x1fS\xd2q%G[\xe9\xad\x12\xb7\xe2\x0f(\xb9<C6x\xda=\x9bo0o\x95M\xe6\xf1\x86\xad\xe6\xe4\xb8\x92\x92s\x8f\xd0\xcc\xc4\xcdq\xd0\x9c2\x99\xcc\xabJ\x0e\x89\xc94\ryF<4(\xabJZ\xa5r:NG\x97\x15\x87&&\x921\x85\x13cJN\x95Q8\xa3&0\xd2\x83\xd3\xb9d>\x9e\x8c1oy\xcf\xf2\x02VdcDUf\x14\xba[VU%\x9fIw\xc9y\xf4\x85\x91\x8d&\xd3\x99|\x0b=\x94H\xc6\x12\xf4\x90\x9c\xa7q%\x9f\x9cL#s\xfc0\xbdY\x87"W\xc6\xb5\xa4\xd3\x99\x1949\xa3\xb4`\xdc\x139%\x9fH\xa6\'i\x9e-\xd9\xd0\xa6jBV\xd9\xa2S\x8a\x9aK\xc6\xe4\xa9\xa9\xc3X\xb2T\x16\xb5\xc6\xb1F\x87\x92j\x02\x1d\xa7\x94<\xdd\xa3\x1c\xa2\xc3\x99\x94\x9c~\xd5\xa3\x87\x82\xb9\x99\xc0\x9c\xd2d*\x9b\xcb\xcc\xf0\x18[\xf3\xb1\x9c\xa2\xa4\xd1\x99\x1c\x97\xc7\x93SI\x15\xad%\xe4\x9c\x1c\xc3\x8ca\xda\x92\xb1<\xcf\x08&\x82f\xe5t\xab\x7f:\x97\xc9*\x18\xe9\x83\xbd\x837\x041@=\x9b\xf9\xcc\xd4\x0czf\xd2iE\x893\x8f\x18\xf6\x8c2\x85J\xe8x*\x93y\x84\xadg"\x93\xc3@\xe3j\xa2uE\xe4\x13\x99\xb4\x8a\xaa\x19*\xc7\xe3\xb8p\xccV&6\x9dbu\xc24\xab\xa5\xe0\xe4X.\x83\xbc\xec\x94\xac\xa2\x95T\xde\x93P\xd5\xec6\xaf\xf7\xd0\xa1C\x1e\xd9(M\x0c+\xe3A\xcb\xde/\xe2\xa9\x87\xb3\x8aQ\x8f\x1c\xb3\x92\x9a\x1a\xc4\xf2\xa7Y\xe9\xa6y}\xd9"F\xfa\x07\xe9P\x16\xf3\x13\xc0\xe0\xa8!\xd0BK\x9d\xd9\xeei7\\`\x1a\x93Y5\xef\xc9\'\xa7<\x99\xdc\xa4w(0\x08=\x90\xc4\xcf\xe6$\xa88\x8e\x80\x02q\xa08d\xc4e\x84b\x90\x81,\x1c\x86\x1c\x97J \x95\x82\x0b\xa9\xcd\xf8\xde\x00m\xd0\x8e\x83B/Je\x90?\x85\xfa\x14\xba\x11\xce\xa1\x16\x9ben7\x03i\xf0\xe0\xe7m\xf7\x97Z\xdb\x80\xd0\xb0\x11E\x1f\xd7nA\xa8\x1f\xf5cha\x10\xf5\xc6\x91\xbb\xd2.\x85\x11NI\xe21\xcb4\'a\x1a\xe3\x90\x91\xb2\x13\xf2\xa8\xa5\xa0L\x9cKPh\xc5\xf1e6\xbe\x8c\xff\x00\x87\xf2\xcb\x9c\r\x18W;\x8e\xcd\xb7\xd5\xfc2\xbbI\xb4Dy\xa6U\xcea\x91\xa6x\xf4\x8f -\x83z_\x94\x0f\x8ar\n\xaf^\x1e9\n\xc7\xe2\xdc*\xb3\x1dB\x89\x11.\x15\xe4\x9a,\x13*\xf7\x96\xe6R\xa3\xb7\xf18\x84\x1e\'P?\xc6+Y\x92\x8cq\xdb\xac#t\xcb\x19\x84\x13FN\x0fb\xbes<\x828\xd7+\xad-\x8f\x9e?[\x81\xdb\xf7\xc6\x08\x8fn\x86\xfb\xdc\xcd\xe9\x0c\xcfs^\x17\xe2yc]z\xceFy\x14\x19\xa4\xb2\\\x1c\xc2H\x98\xdf\x04\x87e\x9e\xcf8\xd7f=\x9664\xc7\xb1\xeb\xe8\x17\xfa\xa1\x86\xael\xd4%\xcd}\xcc\x18Q2\x9d\x16#\xdf\x13|\xces\xbfi\xf4Ay|z\x95o\xf6My\x9ed\x9eu\xbd\xd2)\xe4\xaa\\6\x86\xf4)\xfc\x1d6vY\n\xb3\xa2\xfb\x1a7\xf6\xd1!\xbe+\x13\xc6\x8aS\xdc.\x85=\xf8>\xc4\xbb"\xc3\xeb\x96v\xde\xcdk|#+z\xdfL\x18}J\xb9n\x16\xe1\x0c_E)\x8f\xad\xbc6l%\n\x8f\x94A2\xdf\xf9\xe3\xa81\xc5}\xeb\xb1%xw\xc8\xbc\xb6\x8aQk\x95\xaf\xa0\x94\xaf\xb8\xb1R\x16u\x96SZ\xc1\xcf\xfb\x82\xedw\xc5\xc8\xe9\x83xN\x0c\xde\xd6\xa2\x9e\xc1\x95\xbd\xc9j2\xc5\xe3\xcd\xaf\xb0\x9d\xe6\xd1\xc6\x97\xd7\xa8g\x9bIM\x19\x9e\xf4\x15O\xf1\xf3\xe8\x91\xe5\xfaL\xf0~\xd33\x1a\xe7\xd6Z?\'\xe7\x13<7\xaa\xe15\xc3#\x8a\xe3O\xaf\xb8\xde[\x19\xd4\x9d\xe6\xf5\xd0\xf7\x93\xde\xcd\xeag2\'\xf3\xfcf\x0c\xbd,?\x95T#\x96\x14\xdf\x1f\t\xde\x81Y\xd8\x86\x17K/F\xc7~\x1e\xde\x87+wM\xcc\xd83\x1e#f\xef\xffZ\x8f\xc5\x95\xe5\x19\\\xb9?r\xcb\xb1\xa40\xc6Ac\xf7\xa7\x97w\xdd\xf4\x8a\xfd[\xaa\xc4\x08\x9eA\x83\xfc\xbc\xc8\x1a\xfd\x1302Go\xb1\xc0v\xcd\xadgf;?3o^\x85\xde\x8dI\xc4U\x1eO\x9e\xe7\xd2\xc3\xd70\x89\xfc!\xf40\x08\xc6]\x1c\x96\x9e\xc4\x90n\xf3\xccV\x04w\x8e\x13\x05\x08I\x90IX\r\x0e\x12\x85=d\x0cBd\'t\x10\x1f\xbe\xf1\x83\x16/\xcf!\xd2\x8d8{{H\x07\x14P\xae\x03\xe9\xf7#\xbe\x03\xe9\xdb\xf1\xect\xe0\xdc\x89c\x08\xc7\t\x1c&\x1c\xbaD\x1bJx\xf1\xed5\xf0V\xc4[P\xe3\x02\xce\x84\x0fF\xedD*{\xefB\xbc\x0f\xdf\xbd\xc6;\x80t?\xbe\xfd\x06\xde\x8f8\xbe!J\xca\xf1\x12\xde\xc9\xe7\xf3\xc4\xe4\x9b#\x97\xae\x93\x0b\xd7\t\xbdN\x9e\xf8\x03\t\xfe\x81\x14\xae\x9d\xbc&\xfc\xe7\xd5f\xc7\xd9\xab\xe7\xaf\nCW\xc6\xae\x9c\xbd"\xb6]!\xd6+\xc4\x02\x97m\x97\x83\x97\xa3\x97\xb3\x97_\xba\\Vi\xfd\x90T\xc3\x7f\x90\xba\x7f\xbf\xb4\xc5\xf1~\xc7\xc5\xd0\xbfu\xbc\x17\x82\x8b\xb8\xb2\x8bm\x17\x83\x17\x0b\x17\xb5\x8b\xe6\x8bD\x0c\xbd\'68l\x0bt\xa1m!\xbbPXxg\xe1\xd2\xc2\xd5\x05K\xe1\xad\x93o\t?<\xe7uX\xcf9\xce\t\x8e\xb9\xa1\xb9\'\xe6\xc4\xe8+\xc4\xfa\x8a\xe3\x15!\xf8B\xf4\x05\xe1\xe4\x19b=\xe38\xe3=#>\x7f\xda\xe38\xdd\xbb\xce\xf1\xdc\xb3\xf7:.={\xf5Ya~ia\xee\xd9\x9a\xba\xc092D\x06\xa1\x03s\xb8gN\\r\x9c\xddYOv\xe3\xb2\xac8;pxq\x0c\xe1\xc8\xe08\x81\x03\xbfyP\xdc\x81\xc3K\x06}[\xc4\xb1?\'U\xa7\xec\xa7\xdc\xa7\x1e;u\xec\x949\xfbT\xe1\xa9\x93O\x89\x85\'O>)\x9c\x9d9?#\xe4\x83\xcd\x8eL\xda\xedH\xf7\xfe\xb1\xe3\x8e\x8d\x8d\xa1\xf2\x8db\xa8\x0c\xdd\xa0w_\xffx\x93+\x10\x1d\xf39\xc6Ph\xff\xbe6\xc7\xbe\xdef\xc7\xea\x8d\xabBf\\\xb0\t\x05\xad\xa2C\xec\x14\x87\xc4\x8cxB</\x96[\x86\x83\xeb\x1c{q\\\n^\r\n\xbe`Eu\xc0:\xe4\x18\xf2\x0e\x89\xf3K\x97|\xca\x80\x13\xad\xed\xca\xee*\xec\x12\xfb\x03\xcd\x8e\xbe\xde-\x0ek\xaf\xa3\xd7\xdb{\xa1\xf7\xfd\xde+\xbdec\xbd\xe4E\xfc\x0b\x9c\r\x9c\x0f\x88\xbe@\xb37\xe0\x0b\xacs\x06\xd6\xf6\xd9C\r\x1b\xebCu\xc4\x1a\xb2m\xb4\x86\x04\x82\x85\xde\x08!\xafu\xc9*X\xadc\xd6\'\xac\xa2\x15:A(4\x103\x99\'\'gGG\xdc\xee\x81\xf9\xf2\xa5\xe1\x01\xcd\x12\xdc\xaf\x91\xa3Z\xd3\x08\x9b}{\xf7ieG5\x08\xed\xdb\x1f\x9e%\xe4\xcf"O\x1e?\x0e]w\rh\x1bF\xc2Z\xf4\xae\xc8\x80\x16G\xc0\xc7\x80\x02\x02\xb6\xbbf\x1b\xa0+\x92\xcf\xabn\xfe\x10\xb7\x1b\xe1i\x9c\xc1=\xedF\xe2\x81\xbcN\x85e>\xb8\xf3$\x8fGT\x9e+\x117\x13\xd0q\x82\xb3\x9b\xf1\x90\xc0\xf4\x08j\x1f\xc8\x03\x9b\x18\xd3\xad+1\xed\xbca\x8e+\xeb\x13\x07\x1a\x0f\xfc7CT\x9e\xf3\nendstream\nendobj\n\n6 0 obj\n4161\nendobj\n\n7 0 obj\n<</Type/FontDescriptor/FontName/BAAAAA+LiberationSerif\n/Flags 4\n/FontBBox[-543 -303 1277 981]/ItalicAngle 0\n/Ascent 0\n/Descent 0\n/CapHeight 981\n/StemV 80\n/FontFile2 5 0 R\n>>\nendobj\n\n8 0 obj\n<</Length 222/Filter/FlateDecode>>\nstream\nx\x9c]\x90\xcdj\xc5 \x10\x85\xf7>\xc5,o\x17\x17M\xe82\x04\xca-\x17\xb2\xe8\x0fM\xfb\x00F\'\xb9B3\xca\xc4,\xf2\xf6\x9d\xd8\xb4\x85.\x14\x8e\xe7|zF}\xe9\x1e;\nY\xbfrt=f\x18\x03y\xc6%\xae\xec\x10\x06\x9c\x02\xa9\xaa\x06\x1f\\>T\xd9\xddl\x93\xd2\xc2\xf6\xdb\x92q\xeeh\x8cM\xa3\xf4\x9bxK\xe6\rN\x0f>\x0ex\xa7\xf4\x0b{\xe4@\x13\x9c>.\xbd\xe8~M\xe9\x13g\xa4\x0cF\xb5-x\x1c\xe5\x9e\'\x9b\x9e\xed\x8c\xbaP\xe7\xce\x8b\x1d\xf2v\x16\xe4/\xf0\xbe%\x84\xba\xe8\xea\xbb\x8a\x8b\x1e\x97d\x1d\xb2\xa5\tUcL\x0b\xcd\xf5\xda*$\xff\xcf;\x88at7\xcb\x92\xac$i\xcc}]\xb2\xc7\xe9N\xedc\xfd\xb4\x01\xb72K\x932{\xa9\xb0?\x1e\x08\x7f\xbf\'\xc5\xb4Se}\x01~\x87my\nendstream\nendobj\n\n9 0 obj\n<</Type/Font/Subtype/TrueType/BaseFont/BAAAAA+LiberationSerif\n/FirstChar 0\n/LastChar 1\n/Widths[777 666 ]\n/FontDescriptor 7 0 R\n/ToUnicode 8 0 R\n>>\nendobj\n\n10 0 obj\n<</F1 9 0 R\n>>\nendobj\n\n11 0 obj\n<</Font 10 0 R\n/ProcSet[/PDF/Text]\n>>\nendobj\n\n1 0 obj\n<</Type/Page/Parent 4 0 R/Resources 11 0 R/MediaBox[0 0 595.303937007874 841.889763779528]/Group<</S/Transparency/CS/DeviceRGB/I true>>/Contents 2 0 R>>\nendobj\n\n4 0 obj\n<</Type/Pages\n/Resources 11 0 R\n/MediaBox[ 0 0 595.303937007874 841.889763779528 ]\n/Kids[ 1 0 R ]\n/Count 1>>\nendobj\n\n12 0 obj\n<</Type/Catalog/Pages 4 0 R\n/OpenAction[1 0 R /XYZ null null 0]\n/Lang(de-DE)\n>>\nendobj\n\n13 0 obj\n<</Creator<FEFF005700720069007400650072>\n/Producer<FEFF004C0069006200720065004F0066006600690063006500200037002E0033>\n/CreationDate(D:20240504130543+02\'00\')>>\nendobj\n\nxref\n0 14\n0000000000 65535 f \n0000005198 00000 n \n0000000019 00000 n \n0000000182 00000 n \n0000005367 00000 n \n0000000201 00000 n \n0000004446 00000 n \n0000004467 00000 n \n0000004657 00000 n \n0000004948 00000 n \n0000005111 00000 n \n0000005143 00000 n \n0000005492 00000 n \n0000005589 00000 n \ntrailer\n<</Size 14/Root 12 0 R\n/Info 13 0 R\n/ID [ <74E7A94EB7CDF6C42C6B8B35D5DC56B6>\n<74E7A94EB7CDF6C42C6B8B35D5DC56B6> ]\n/DocChecksum /F25ABF5E9B9479C018E28D8071D25F67\n>>\nstartxref\n5764\n%%EOF\n'
    """

    a_path = tmp_path / "A.pdf"
    b_path = tmp_path / "B.pdf"
    out_path = tmp_path / "out.pdf"

    with open(a_path, "wb") as f:
        f.write(pdfA_content)
    with open(b_path, "wb") as f:
        f.write(pdfB_content)

    p.overlay(str(b_path), str(a_path), str(out_path))

    pdf_out = pypdf.PdfReader(out_path)
    page_out = pdf_out.pages[0]

    assert p.page_count(str(out_path)) == 1
    assert page_out.extract_text() == 'BA'

    p.overlay(str(a_path), str(b_path), str(out_path))

    pdf_out = pypdf.PdfReader(out_path)
    page_out = pdf_out.pages[0]

    assert page_out.extract_text() == 'AB'


#testing layering with transparency
def test_overlay_layering(tmp_path):
    yellow_bytes = b"""
    b"%PDF-1.6\n%\xc3\xa4\xc3\xbc\xc3\xb6\xc3\x9f\n2 0 obj\n<</Length 3 0 R/Filter/FlateDecode>>\nstream\nx\x9ce\x8e\xcb\n\x021\x0cE\xf7\xf9\x8a\xbb\x16\xcc\xa4M\xd3i\xf7\xca\x80\xbbq\x04\xbf@F\xc4\x07\x8e\x8b\xf9}[\x1f\x0bq\x11\x92\x03\x87{#\xec0\xd3\x1d\x02a\xf1\t\x96\x8d}kH\xc1q\x8a\x0e\xd3\x81\xf6\x0b\\\xc9\xc1\x15g\x1a\x8b\xda\xac\xbb\xc10>\xd0\xec\xa6\x80\xd5\r=\t{\xd1\xec\x9d\xc5\x14K\x90\xe6(m\n\xea\xb4@\xac\xbb\xf5\x162\xb6\xdd\xabi.\xb3)y'RQ\x0eV\xabS\xc6\x85\x96\xe5\xf8\xd2\x19o\xaa\x8fd\xad\x1c%r\xfe\xe3\x8f\xfd\x1bu\xc6\x91\x06\xea\xa9\xc7\x13\x94\x18.\x01\nendstream\nendobj\n\n3 0 obj\n154\nendobj\n\n4 0 obj\n<</Type/XObject\n/Subtype/Form\n/BBox[ -0.05 0.039 606.95 841.94 ]\n/Resources 6 0 R\n/Group<</S/Transparency/CS/DeviceRGB/K true>>\n/Length 60\n/Filter/FlateDecode\n>>\nstream\nx\x9c360\xd631U0\xd03\xb0\xb0T\xc8\xe5\xd2\x052`\xbc\x1c\x05\x08\xcf\xc2\xc4P\xcf\xd2\x18\xc4730\xd3\xb3\xc4\xe0CUs\x19#\x1b\x95\xa3\x90\xc1\xa5\x90\xa6\xc5\x05\x00\x8e\xd2\x12\xe6\nendstream\nendobj\n\n5 0 obj\n<</CA 0.5\n   /ca 0.5\n>>\nendobj\n\n8 0 obj\n<<\n>>\nendobj\n\n6 0 obj\n<</Font 8 0 R\n/XObject<</Tr4 4 0 R>>\n/ExtGState<</EGS5 5 0 R>>\n/ProcSet[/PDF/Text/ImageC/ImageI/ImageB]\n>>\nendobj\n\n1 0 obj\n<</Type/Page/Parent 7 0 R/Resources 6 0 R/MediaBox[0 0 595.303937007874 841.889763779528]/Group<</S/Transparency/CS/DeviceRGB/I true>>/Contents 2 0 R>>\nendobj\n\n7 0 obj\n<</Type/Pages\n/Resources 6 0 R\n/MediaBox[ 0 0 595.303937007874 841.889763779528 ]\n/Kids[ 1 0 R ]\n/Count 1>>\nendobj\n\n9 0 obj\n<</Type/Catalog/Pages 7 0 R\n/OpenAction[1 0 R /XYZ null null 0]\n/Lang(de-DE)\n>>\nendobj\n\n10 0 obj\n<</Creator<FEFF005700720069007400650072>\n/Producer<FEFF004C0069006200720065004F0066006600690063006500200037002E0033>\n/CreationDate(D:20240506140754+02'00')>>\nendobj\n\nxref\n0 11\n0000000000 65535 f \n0000000705 00000 n \n0000000019 00000 n \n0000000244 00000 n \n0000000264 00000 n \n0000000520 00000 n \n0000000582 00000 n \n0000000873 00000 n \n0000000560 00000 n \n0000000997 00000 n \n0000001093 00000 n \ntrailer\n<</Size 11/Root 9 0 R\n/Info 10 0 R\n/ID [ <2C2195FD8D256CF944D60F79A3C88DBD>\n<2C2195FD8D256CF944D60F79A3C88DBD> ]\n/DocChecksum /4B42A3E875E00B844470F45DDB22C581\n>>\nstartxref\n1268\n%%EOF\n"
    """

    blue_bytes = b"""
    b'%PDF-1.6\n%\xc3\xa4\xc3\xbc\xc3\xb6\xc3\x9f\n2 0 obj\n<</Length 3 0 R/Filter/FlateDecode>>\nstream\nx\x9ce\x8eK\x0b\xc2@\x0c\x84\xef\xf9\x15s\x16\xdcf\x1fIw\xefJ\xc1[\xad\xe0/\x90\x8a\xf8\xc0z\xe8\xdf7\xeb\xe3 \x1eB\xf2\x91af\xd8y\xcct\x07\x83\x1d\x87\x0c)\xe2B+\xc8\xc9\xbb\xac\x1e\xd3\x81\xf6\x0b\\\xc9\xfe0\x1aM\xda\xac\xbbA0>\xd0\xec\xa6\x84\xd5\r=\xb1\x0b\x1cK\xf0\xa2Y\xcd(\x16\xe56\xa7\xe8\xa3\x81\xd6\xdd\x06I\x05\xdb\xee\x954\xdbl\xcc\xedD\x91\xa3KR\xa3s\xc1\x85\x96v|\xe9\x8c7\xd5"%VVVW\xfe\xf8\xa3\xfe\xb5:\xe3H\x03\xf5\xd4\xe3\t\x93^.\x00\nendstream\nendobj\n\n3 0 obj\n153\nendobj\n\n4 0 obj\n<</Type/XObject\n/Subtype/Form\n/BBox[ -0.05 0.039 606.95 841.94 ]\n/Resources 6 0 R\n/Group<</S/Transparency/CS/DeviceRGB/K true>>\n/Length 60\n/Filter/FlateDecode\n>>\nstream\nx\x9c360\xd631U0\xd03\xb0\xb0T\xc8\xe5\xd2\x052`\xbc\x1c\x05\x08\xcf\xc2\xc4P\xcf\xd2\x18\xc4730\xd3\xb3\xc4\xe0CUs\x19#\x1b\x95\xa3\x90\xc1\xa5\x90\xa6\xc5\x05\x00\x8e\xd2\x12\xe6\nendstream\nendobj\n\n5 0 obj\n<</CA 0.5\n   /ca 0.5\n>>\nendobj\n\n8 0 obj\n<<\n>>\nendobj\n\n6 0 obj\n<</Font 8 0 R\n/XObject<</Tr4 4 0 R>>\n/ExtGState<</EGS5 5 0 R>>\n/ProcSet[/PDF/Text/ImageC/ImageI/ImageB]\n>>\nendobj\n\n1 0 obj\n<</Type/Page/Parent 7 0 R/Resources 6 0 R/MediaBox[0 0 595.303937007874 841.889763779528]/Group<</S/Transparency/CS/DeviceRGB/I true>>/Contents 2 0 R>>\nendobj\n\n7 0 obj\n<</Type/Pages\n/Resources 6 0 R\n/MediaBox[ 0 0 595.303937007874 841.889763779528 ]\n/Kids[ 1 0 R ]\n/Count 1>>\nendobj\n\n9 0 obj\n<</Type/Catalog/Pages 7 0 R\n/OpenAction[1 0 R /XYZ null null 0]\n/Lang(de-DE)\n>>\nendobj\n\n10 0 obj\n<</Creator<FEFF005700720069007400650072>\n/Producer<FEFF004C0069006200720065004F0066006600690063006500200037002E0033>\n/CreationDate(D:20240506140832+02\'00\')>>\nendobj\n\nxref\n0 11\n0000000000 65535 f \n0000000704 00000 n \n0000000019 00000 n \n0000000243 00000 n \n0000000263 00000 n \n0000000519 00000 n \n0000000581 00000 n \n0000000872 00000 n \n0000000559 00000 n \n0000000996 00000 n \n0000001092 00000 n \ntrailer\n<</Size 11/Root 9 0 R\n/Info 10 0 R\n/ID [ <2B7F217A54418EE092E1F5D1C04A57BB>\n<2B7F217A54418EE092E1F5D1C04A57BB> ]\n/DocChecksum /BD76F989EA01A4D0B9FED381355337F4\n>>\nstartxref\n1267\n%%EOF\n'
    """

    blue_yellow_bytes = b'%PDF-1.3\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<\n/Type /Pages\n/Count 1\n/Kids [ 4 0 R ]\n>>\nendobj\n2 0 obj\n<<\n/Producer (pypdf)\n>>\nendobj\n3 0 obj\n<<\n/Type /Catalog\n/Pages 1 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Page\n/Resources <<\n/ExtGState <<\n/EGS5 5 0 R\n/EGS5-0 6 0 R\n>>\n/XObject <<\n/Tr4 7 0 R\n/Tr4-0 10 0 R\n>>\n/ProcSet [ /ImageB /ImageC /ImageI /PDF /Text ]\n>>\n/MediaBox [ 0.0 0.0 595.303937 841.889764 ]\n/Group <<\n/S /Transparency\n/CS /DeviceRGB\n/I true\n>>\n/Contents 13 0 R\n/Rotate 0\n/Annots [ ]\n/Parent 1 0 R\n>>\nendobj\n5 0 obj\n<<\n/CA 0.5\n/ca 0.5\n>>\nendobj\n6 0 obj\n<<\n/CA 0.5\n/ca 0.5\n>>\nendobj\n7 0 obj\n<<\n/Type /XObject\n/Subtype /Form\n/BBox [ -0.05 0.039 606.95 841.94 ]\n/Resources 8 0 R\n/Group <<\n/S /Transparency\n/CS /DeviceRGB\n/K true\n>>\n/Filter /FlateDecode\n/Length 60\n>>\nstream\nx\x9c360\xd631U0\xd03\xb0\xb0T\xc8\xe5\xd2\x052`\xbc\x1c\x05\x08\xcf\xc2\xc4P\xcf\xd2\x18\xc4730\xd3\xb3\xc4\xe0CUs\x19#\x1b\x95\xa3\x90\xc1\xa5\x90\xa6\xc5\x05\x00\x8e\xd2\x12\xe6\nendstream\nendobj\n8 0 obj\n<<\n/Font 9 0 R\n/XObject <<\n/Tr4 7 0 R\n>>\n/ExtGState <<\n/EGS5 5 0 R\n>>\n/ProcSet [ /PDF /Text /ImageC /ImageI /ImageB ]\n>>\nendobj\n9 0 obj\n<<\n>>\nendobj\n10 0 obj\n<<\n/Type /XObject\n/Subtype /Form\n/BBox [ -0.05 0.039 606.95 841.94 ]\n/Resources 11 0 R\n/Group <<\n/S /Transparency\n/CS /DeviceRGB\n/K true\n>>\n/Filter /FlateDecode\n/Length 60\n>>\nstream\nx\x9c360\xd631U0\xd03\xb0\xb0T\xc8\xe5\xd2\x052`\xbc\x1c\x05\x08\xcf\xc2\xc4P\xcf\xd2\x18\xc4730\xd3\xb3\xc4\xe0CUs\x19#\x1b\x95\xa3\x90\xc1\xa5\x90\xa6\xc5\x05\x00\x8e\xd2\x12\xe6\nendstream\nendobj\n11 0 obj\n<<\n/Font 12 0 R\n/XObject <<\n/Tr4 10 0 R\n>>\n/ExtGState <<\n/EGS5 6 0 R\n>>\n/ProcSet [ /PDF /Text /ImageC /ImageI /ImageB ]\n>>\nendobj\n12 0 obj\n<<\n>>\nendobj\n13 0 obj\n<<\n/Length 515\n>>\nstream\nq\nq\n1 0.0 0.0 1 0.0 0.0  cm\n0.1 w\nq\n0 0.028 595.275 841.861 re\nW*\nn\n0 0 1 rg\nq\n/EGS5 gs\n/Tr4 Do\nQ\n0.20392157 0.39607843 0.64313725 RG\nq\n0 w\n0 J\n1 j\n303.45 0.089 m\n-0.05 0.089 l\n-0.05 841.939 l\n606.9 841.939 l\n606.9 0.089 l\n303.45 0.089 l\nh\nS\nQ\nQ\nQ\nQ\n\nq\n0.0 0.0 595.303937 841.889764 re\nW\nn\n0.1 w\nq\n0 0.028 595.275 841.861 re\nW*\nn\n1 1 0 rg\nq\n/EGS5-0 gs\n/Tr4-0 Do\nQ\n0.20392157 0.39607843 0.64313725 RG\nq\n0 w\n0 J\n1 j\n303.45 0.089 m\n-0.05 0.089 l\n-0.05 841.939 l\n606.9 841.939 l\n606.9 0.089 l\n303.45 0.089 l\nh\nS\nQ\nQ\nQ\n\n\nendstream\nendobj\nxref\n0 14\n0000000000 65535 f \n0000000015 00000 n \n0000000074 00000 n \n0000000113 00000 n \n0000000162 00000 n \n0000000493 00000 n \n0000000530 00000 n \n0000000567 00000 n \n0000000834 00000 n \n0000000970 00000 n \n0000000991 00000 n \n0000001260 00000 n \n0000001399 00000 n \n0000001421 00000 n \ntrailer\n<<\n/Size 14\n/Root 3 0 R\n/Info 2 0 R\n>>\nstartxref\n1988\n%%EOF\n'

    y_path = tmp_path / "yellow.pdf"
    b_path = tmp_path / "blue.pdf"
    out_path = tmp_path / "out.pdf"

    with open(y_path, "wb") as f:
        f.write(yellow_bytes)
    with open(b_path, "wb") as f:
        f.write(blue_bytes)

    p.overlay(str(b_path), str(y_path), str(out_path))

    with open(str(out_path),"rb") as f:
        assert f.read() == blue_yellow_bytes