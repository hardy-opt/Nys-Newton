import mlreportgen.dom.*;
d = Document('myreport','pdf');
open(d); 

t = Table(magic(5));
t.Style = {RowHeight('1in')};
t.Border = 'solid';
t.BorderWidth = '1px';
t.ColSep = 'solid';
t.ColSepWidth = '1';
t.RowSep = 'solid';
t.RowSepWidth = '1';

% Set this property first to prevent overwriting alignment properties
t.TableEntriesStyle = {FontFamily('Arial'),Width('1in')};
    %,Color('red'),Bold};
t.TableEntriesHAlign = 'center';
t.TableEntriesVAlign = 'middle';


%%%%


%d = Document('test','pdf');

%m = magic(5);

 x = min(min(m(1,:)));
[v,i] = find(m==x);
v
i
y = min(min(m(2,:)));
[q,p] =find(m==y);
q
p
%[q1,p1] = (min(m(:,3)));
t = Table(m);
t.Border = 'single';
t.ColSep = 'single';
t.RowSep = 'single';

t.TableEntriesInnerMargin = '2pt';
t.TableEntriesHAlign = 'right';

maxnum = entry(t,v,i);
maxnum.Children(1).Color = 'Red';


maxnum = entry(t,q,p);
maxnum.Children(1).Color = 'Red';


%%%%



append(d,t);
%close(d);
%open(d);
%rptview(d.OutputPath);
t1 = Table(magic(6));
t1.Style = {RowHeight('1in')};
t1.Border = 'solid';
t1.BorderWidth = '1px';
t1.ColSep = 'solid';
t1.ColSepWidth = '1';
t1.RowSep = 'solid';
t1.RowSepWidth = '1';

% Set this property first to prevent overwriting alignment properties
t1.TableEntriesStyle = {FontFamily('Arial'),Width('1in'),Color('red'),Bold};
t1.TableEntriesHAlign = 'center';
t1.TableEntriesVAlign = 'middle';

append(d,t1);
close(d);

rptview(d.OutputPath);