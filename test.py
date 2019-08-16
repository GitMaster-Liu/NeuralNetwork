import xlrd
import xlwt
import math



def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def doubleS1(x):
    output = 10*(2+math.exp(10*(2.3-x)))/(1+math.exp(10*(-2.3-x)))/(1+math.exp(10*(2.3-x)))
    return output

def doubleS2(x):
    output = 10*(2+math.exp(10*(4.3-x)))/(1+math.exp(10*(-4.3-x)))/(1+math.exp(10*(4.3-x)))
    return output

def doubleS3(x):
    output = 10*(2+math.exp(10*(1.8-x)))/(1+math.exp(10*(-1.8-x)))/(1+math.exp(10*(1.8-x)))
    return output


ExcelFile=xlrd.open_workbook(r'C:\Users\lenovo\Desktop\data1800.xls')


sheet1=ExcelFile.sheet_by_index(0)
sheet2=ExcelFile.sheet_by_index(1)
f = xlwt.Workbook()
sheet4 = f.add_sheet('Sheet4', cell_overwrite_ok=True)
style = set_style('Times New Roman', 220, True)
for n in range(1800):
    row = sheet1.row_values(n)
    minus=[10, 18, 18, 4, 4, 4]
    x_test2=[0, 0, 0, 0, 0, 0]
    print(row)
    for i in range(len(row)):
        x_test2[i] = row[i] - minus[i]
    x_test2[0] = doubleS1(x_test2[0])
    x_test2[1] = doubleS2(x_test2[1])
    x_test2[2] = doubleS2(x_test2[2])
    x_test2[3] = doubleS3(x_test2[3])
    x_test2[4] = doubleS3(x_test2[4])
    x_test2[5] = doubleS3(x_test2[5])
    print(x_test2)
    for j in range(0, len(x_test2)):
        sheet4.write(n, j, x_test2[j], style)
f.save(r'C:\Users\lenovo\Desktop\datatezheng1.xlsx')