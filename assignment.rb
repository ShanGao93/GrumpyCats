require 'json'

module Assignment
    def assignment?
        true
    end
    
    def mean(x)
        return x.reduce(0.0) do |u, p|
            u += p.to_f/x.length 
        end
    end

    def std(x)
        m = mean x
        l = (x.length-1).to_f
        u = x.reduce(0.0) do |u, p|
            u += p.nil? ? 0 : ((p-m)**2 / l)
        end
        return Math.sqrt u
    end
    
    def kcross_examination x, k, m, p
        s = []
        r = x.shuffle.each_slice((x.length.to_f/k.to_f).ceil).to_a
        (0..k-1).each do |i|
            t = r[i]
            n = []
            r.each.with_index do |p, j| n += p if j !=i end
            c = m.train(n, p)
            s << model.loss(t, c)
        end
        return { "eta" => eta, "avg" => (mean s), "std" => (std s) }
    end
    
    def update_weights(w, dw, learning_rate)
        w1 = w.clone
        dw.each_key do |k|
            w1[k] -= learning_rate * dw[k]
        end
        w1
    end
    
    def norm w
        0.5 * Math.sqrt(w.keys.inject(0.0) {|u,k| u += w[k] ** 2.0})
    end
    
    def gradient_descent x, w, model, learning_rate = 1e-4, rmse_tol = 1e-3, max_iter = 1000  
        # normalization
        data = x
        rm_prev = model.func data, w
        diff = 1000.0
        
        iters = [0]
        rmses = [rm_prev]
        norms = [norm(w)]
        
        (1..max_iter).each do |i|
            break if diff <= rmse_tol
            
            dw = model.grad x, w
            w = update_weights w, dw, learning_rate
            model.adjust w
            rm = model.func data, w
            diff = rm_prev - rm
            rm_prev = rm
            
            norms << norm(w)
            iters << i
            rmses << rm
        end
        return [iters, rmses, norms, w, data]
    end
    
    def plot x, y
        Daru::DataFrame.new({x: x, y: y}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
            plot.x_label "X"
            plot.y_label "Y"
        end
    end
    
    def cross_validate x, folds, &block
      fold_size = x.size / folds
      subsets = []
      x.shuffle.each_slice(fold_size) do |subset|
        subsets << subset
      end
      i_folds = Array.new(folds) {|i| i}

      folds.times do |fold|
        test = subsets[fold]
        train = (i_folds - [fold]).flat_map {|t_fold| subsets[t_fold]}
        yield train, test, fold
      end
    end
    
    def confusion_matrix cls_names, x, predictions
      counts = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0}}
      cls_names.each {|i| cls_names.each{|j| counts[i][j] = 0}}

      x.each.with_index do |row, i|
        predicted_label = predictions[i].keys.first
        actual_label = row["label"]

        counts[predicted_label.to_s][actual_label.to_s] += 1
      end

      return counts
    end
    
    def accuracy conf_mat
      correct = 0.0
      sum = 0.0

      conf_mat.each_key do |pred|
        conf_mat[pred].each_key do |act|
          sum += conf_mat[pred][act]
          correct += conf_mat[pred][act] if pred == act
        end
      end

      correct / sum
    end
    
    def print_matrix_cell t="", s = 25, p = "l"
        return t if t.to_s.length >= s
        
        p == "m" ? l = " " * (( s - t.to_s.length ) / 2) + t.to_s + " " * (( s - t.to_s.length ) / 2) :
            l = t.to_s + " " * ( s - t.to_s.length )
        print l
    end
    
    def print_matrix_row t=[], s = 25, p = "l"
        t.each do |ti| print_matrix_cell ti, s, p end
        puts
    end
    
    # turns data to data_class
    def parse_class x
        s = Hash.new {|h,k| h[k] = 0}
        x.each do |p|
            s[p["label"]] += 1
        end
        return s
    end
    
    # Binary split on categorical features
    def parse_category x, col
        s = Hash.new {|h,k| h[k] = 0}
        x.each do |p|
            s[p[col]] += 1
        end
        return s
    end
    
    def p_log_p(x)
        return ( x == 0 ? 0 : x * Math.log(x,2) )
    end
    
    def entropy p
        total = total_instance(p)
        return -p.values.reduce(0.0) do |u, pi|
            u += p_log_p(pi/total)
        end
    end
    
    # x is the entropy of total instances, n is the number of entropy
    def information_gain x, n, splits
        return splits.reduce(x) do |u, s|
            u -= total_instance(s)/n*entropy(s)
        end
    end
    
    def total_instance s
        return s.values.reduce(0.0) do |t, p|
            t += p
        end
    end
    
    def best_ig data, feature, type
        max_ig = 0
        best_v = 0
        c = parse_class data
        e = entropy(c)
        n = data.length
        left = Hash.new {|h,k| h[k] = 0}
        right = parse_class data
        
        groups = data.group_by{|h| h[feature]}
        
        if type == "TEXT"
            # categorical features
            groups.each do |p|
                p[1].each do |pi|
                    left[pi["label"]] += 1
                    right[pi["label"]] -= 1
                end
                
                ig = information_gain(e, n, [left, right])
                
                if ig > max_ig
                    max_ig = ig
                    best_v = p[0]
                end
                left = Hash.new {|h,k| h[k] = 0}
                right = parse_class data
            end
        else
            # numericals goes here
            groups.each do |p|
                ig = information_gain(e, n, [left, right])
                p[1].each do |pi|
                    left[pi["label"]] += 1
                    right[pi["label"]] -= 1
                end
                
                if ig > max_ig
                    max_ig = ig
                    best_v = p[0]
                end
            end
        end
        
        return [max_ig, best_v]
    end
    
    # dot mulitply
    def dot row, w
      f = row["features"]
      u = 0.0
      f.keys.each do |k|
        u += f[k] * w[k] if (f[k] != "" && !f[k].nil?)
      end
      return u
    end
    
    # Returns the squared error
    def error(x, w)
        u = 0.0
        x.each do |p|
            yi = dot(p, w)
            u += 0.5*(yi - p["label"])**2
        end
        return u
    end
    
    def cross_validate x, folds, &block
        fold_size = x.size / folds
        subsets = []
        x.shuffle.each_slice(fold_size) do |subset|
            subsets << subset
        end
        i_folds = Array.new(folds) {|i| i}
        
        folds.times do |fold|
            test = subsets[fold]
            train = (i_folds - [fold]).flat_map {|t_fold| subsets[t_fold]}
            yield train, test, fold
        end
    end
    
    def confusion_matrix cls_names, x, predictions
        counts = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0}}
        cls_names.each {|i| cls_names.each{|j| counts[i][j] = 0}}
        
        x.each.with_index do |row, i|
            predicted_label = predictions[i].keys.first
            actual_label = row["label"]
            
            counts[predicted_label.to_s][actual_label.to_s] += 1
        end
        
        return counts
    end
    
    def accuracy conf_mat
        correct = 0.0
        sum = 0.0
        
        conf_mat.each_key do |pred|
            conf_mat[pred].each_key do |act|
                sum += conf_mat[pred][act]
                correct += conf_mat[pred][act] if pred == act
            end
        end
        
        correct / sum
    end
    
    
    def print_matrix_cell t="", s = 25, p = "l"
        return t if t.to_s.length >= s
        
        p == "m" ? l = " " * (( s - t.to_s.length ) / 2) + t.to_s + " " * (( s - t.to_s.length ) / 2) :
            l = t.to_s + " " * ( s - t.to_s.length )
        print l
    end
    
    def print_matrix_row t=[], s = 25, p = "l"
        t.each do |ti| print_matrix_cell ti, s, p end
        puts
    end
    
    def mean(x)
        return x.reduce(0.0) do |u, p|
            u += p.to_f/x.length 
        end
    end
    
    def std(x)
        m = mean x
        l = (x.length-1).to_f
        u = x.reduce(0.0) do |u, p|
            u += p.nil? ? 0 : ((p-m)**2 / l)
        end
        return Math.sqrt u
    end
    
    def roc_curve train, predictions, scores
        tprs = [0.0]
        fprs = [0.0]
        
        scores.each do |s|
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            train.each.with_index do |row, i|
                y = row["label"]
                p = (predictions[i].values[0] > s) ? 1 : 0
                if y == p
                    tp += 1 if p.to_i.to_s == "1"
                    tn += 1 if p.to_i.to_s == "0"
                else
                    fp += 1 if p.to_i.to_s == "1"
                    fn += 1 if p.to_i.to_s == "0"
                end
            end
            tpr = (tp + fn) == 0.0 ? 0.0 : tp.to_f / (tp + fn) 
            fpr = (fp + tn) == 0.0 ? 0.0 : fp.to_f / (fp + tn)
            tprs << tpr
            fprs << fpr
        end
        
        # Add the last iteration, so the maximum auc can be 1.0
        tprs << 1.0
        fprs << 1.0
        
        return fprs, tprs
    end
    
    def auc fprs, tprs
        x = 0.0
        y = 0.0
        area = 0.0
        fprs.each.with_index do |fpr, i|
            tpr = tprs[i]
            area += (fpr - x) * (tpr + y) / 2.0
            x = fpr
            y = tpr
        end
        return area
    end
end